"""Main Agent4LR agent loop: multi-agent LR pipeline with prefix-reusable checkpoints.

An LR run is driven by an ``LRConfig`` — an ordered chain of
``AgentSpec``s (v1: planner → tool_caller → finisher, exactly one
AgentSpec per slot). The runner executes the chain slot-by-slot, writing
a content-addressable checkpoint after each completed slot. Subsequent
runs whose configurations share a leading prefix reuse those checkpoints
verbatim — only the diverging suffix incurs new LLM calls.

The per-API-call slot bodies live in :mod:`src.agent4lr.stepper`
(``build_request`` / ``apply_response``); this runner drives them
synchronously, while the batch workflow drives the same functions
asynchronously via the OpenAI Batch API.

Slot semantics:

* **Planner** — single LLM call. No tools (``tool_choice="none"``);
  produces a free-text plan. Input: system prompt + initial user prompt
  (bug report + cleaned trigger test + numbered candidate list).
* **Tool loop** — up to ``AgentSpec.iterations`` ReAct iterations. Tools:
  ``get_snippet_of_method(method_number: int)`` and ``exit()``
  (``tool_choice="required"``). Terminates on ``exit()`` or budget
  exhaustion.
* **Finisher** — single LLM call forced to emit
  ``rank_methods(top_5_methods: int[5])``
  (``tool_choice="required"``). The structured arguments map directly
  to candidate indices.

Context-overflow handling is **deferred** — providers
raise ``ContextOverflowError`` but the runner does not catch it. A
future patch can wrap this function with a decrement-and-retry loop
without changing the public API.

Transcript convention: the runner maintains **two parallel lists**:

* ``input_list`` — canonical OpenAI Responses-API-native items
  (reasoning, message, function_call, function_call_output, plus
  role/content). This is what each provider's ``chat(history=...)``
  consumes; the OpenAI backend passes it straight to ``responses.create``,
  the Ollama backend projects it to chat-completions internally
  (dropping reasoning, folding function_call/output pairs onto
  assistant tool_calls + tool-role outputs).
* ``messages`` — human-readable view mirroring FlexFL's
  ``state.messages``: assistant tool calls render as
  ``"[Function call: name(arg1, arg2)]"`` strings; tool outputs are
  ``role: "user"`` messages. Includes the post-``rank_methods``
  ``Top_N : <fqn>`` block that the reference adds to the readable
  transcript but not to the API input.

Both lists are persisted in ``lr_result.json`` (schema_version=2) and
in each completed-slot checkpoint.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.checkpoints import CheckpointStore, CompletedCheckpoint, PendingSlotState
from src.agent4lr.configs import validate_v1_invariant
from src.agent4lr.io import LRBugInputs, load_lr_bug_inputs
from src.agent4lr.providers import Provider, build_provider
from src.agent4lr.stepper import (
    apply_response,
    build_request,
    extract_rank_methods_call,
    initial_state,
    slot_is_complete,
)
from src.agent4lr.tools import LRToolContext, rank_methods
from src.common.config import get_lr_checkpoints_dir, get_lr_model_dir, get_processed_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases and constants
# ---------------------------------------------------------------------------

LRConfig = tuple[AgentSpec, ...]

LR_RESULT_FILE = "lr_result.json"
TOP5_FILE = "top5.txt"

DEFAULT_LR_INPUT_KEYS: tuple[str, ...] = ("bug_report", "trigger_test")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LRRunResult:
    """Result of a single Agent4LR run for one bug.

    ``agent_chain`` is the list of ``AgentSpec.identity()`` dicts (one
    per executed slot). ``top5`` are the final FQNs (out-of-range
    indices were dropped with a WARN; the list may be shorter than 5).

    ``input_list`` is the canonical OpenAI Responses-API-native
    conversation state (reasoning + function_call + function_call_output
    items, plus role/content); ``messages`` is the parallel
    human-readable view (assistant ``"[Function call: ...]"`` strings,
    ``role:"user"`` tool-output messages, post-rank ``Top_N`` block).
    ``response_dumps`` is the per-call normalised provider payloads.
    """

    project: str
    bug_id: str
    config_name: str
    agent_chain: list[dict[str, Any]]
    sr_model_id: str
    candidate_source: str
    started_at: float
    finished_at: float
    top5_indices: list[int]
    top5: list[str]
    input_list: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    response_dumps: list[dict[str, Any]]
    input: list[str] = field(default_factory=list)
    schema_version: int = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_input_keys(requested: tuple[str, ...], inputs: LRBugInputs) -> list[str]:
    """Intersection of requested input keys with what's present on disk."""
    effective: list[str] = []
    if "bug_report" in requested and inputs.has_bug_report():
        effective.append("bug_report")
    if "trigger_test" in requested and inputs.has_trigger_test():
        effective.append("trigger_test")
    return effective


def _provider_for(
    spec: AgentSpec, *, ollama_base_url: str, ollama_verify: bool, openai_api_key: str | None
) -> Provider:
    return build_provider(
        provider=spec.provider,
        model=spec.model,
        base_url=ollama_base_url,
        verify=ollama_verify,
        temperature=spec.temperature,
        top_p=spec.top_p,
        reasoning_effort=spec.reasoning_effort,
        api_key=openai_api_key,
        **(spec.provider_opts or {}),
    )


def _extract_top5_from_cached_input_list(
    input_list: list[dict[str, Any]],
) -> list[int]:
    """Walk a cached ``input_list`` for the finisher's rank_methods arguments.

    Looks for the most recent ``{"type":"function_call",
    "name":"rank_methods"}`` item and JSON-parses its ``arguments``.
    """
    for item in reversed(input_list):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") != "rank_methods":
            continue
        args_raw = item.get("arguments") or "{}"
        args: Any
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
        else:
            args = args_raw
        if isinstance(args, dict):
            return extract_rank_methods_call([{"name": "rank_methods", "arguments": args}])
    return []


# ---------------------------------------------------------------------------
# Skip-on-no-fault helpers
# ---------------------------------------------------------------------------


def _load_fault_signatures(processed_dir: Path) -> set[str]:
    """Return the set of corpus-id signatures from ``faults.csv``.

    Returns an empty set when the file is missing or has no non-blank
    signatures. The runner uses this to decide whether the LR phase
    has any chance of recovering the answer.
    """
    csv_path = processed_dir / "faults.csv"
    if not csv_path.exists():
        return set()
    sigs: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sig = (row.get("signature") or "").strip()
            if sig:
                sigs.add(sig)
    return sigs


def should_skip_no_fault(
    *,
    project: str,
    bug_id: str | int,
    candidates: list[str],
    dataset: str = "defects4j",
) -> bool:
    """True when LR has no chance of recovering the fault for this bug.

    Combines :func:`_load_fault_signatures` and
    :func:`_any_fault_in_candidates` — the same gate the sync runner
    applies before any LLM work (skip means: write nothing, keeping the
    evaluation framework's "missing = excluded from denominator"
    semantics correct). Public so the batch workflow can apply it before
    paying for batch requests.
    """
    processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
    fault_sigs = _load_fault_signatures(processed_dir)
    return not _any_fault_in_candidates(candidates, fault_sigs)


def candidate_list_skip_reason(candidates: list[str]) -> str | None:
    """Reason to skip when the SR top-20 is not a valid FlexFL candidate list.

    FlexFL's combination rule (``SBIR[:5] + Ochiai[:5] + BoostN[:5] +
    Agent4SR[:5]``, duplicates allowed) always yields exactly 20 lines when
    every source contributed 5. Anything else means a source was short or
    missing, so the universe is not comparable across bugs — refuse to run.
    Returns ``None`` when the list is valid. Public so both the sync runner
    and the batch tick apply the same gate.
    """
    if len(candidates) != 20:
        return f"top-20 invalid ({len(candidates)} lines, need exactly 20)"
    return None


def _any_fault_in_candidates(candidates: list[str], fault_signatures_corpus: set[str]) -> bool:
    """True if any fault signature appears in the SR top-20 candidate list.

    ``faults.csv`` uses corpus-id form (``pkg$Cls.method(P)``); the
    top-20 file uses dotted form (``pkg.Cls.method(P)``). Normalise the
    fault side via ``$ → .`` (mirroring :mod:`src.evaluation.sources`)
    and union-check against the candidates set. Also accepts an exact
    corpus-id match in case both sides happen to be in the same form.
    """
    if not fault_signatures_corpus or not candidates:
        return False
    candidates_set = set(candidates)
    fault_dotted = {s.replace("$", ".") for s in fault_signatures_corpus}
    return bool(fault_dotted & candidates_set or fault_signatures_corpus & candidates_set)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_agent4lr_for_bug(
    *,
    project: str,
    bug_id: str | int,
    config_name: str,
    config: LRConfig,
    sr_model_id: str,
    input_keys: tuple[str, ...] = DEFAULT_LR_INPUT_KEYS,
    dataset: str = "defects4j",
    lr_model_id: str | None = None,
    ollama_base_url: str = "",
    ollama_verify: bool = True,
    openai_api_key: str | None = None,
) -> LRRunResult | None:
    """Run the full Agent4LR pipeline for one bug.

    See the module docstring for the slot semantics and the cross-config
    checkpoint reuse algorithm.

    Returns ``None`` when both ``lr_result.json`` and ``top5.txt``
    already exist for ``lr_model_id`` (skip-on-existing). Otherwise
    returns the fresh :class:`LRRunResult`. Raises
    :class:`ValueError` on invariant violation,
    :class:`FileNotFoundError` if required inputs are missing,
    :class:`src.agent4lr.providers.ContextOverflowError` (uncaught; see
    module docstring).
    """
    validate_v1_invariant(config)
    bug_id_str = str(bug_id)

    inputs = load_lr_bug_inputs(
        project, bug_id, sr_model_id=sr_model_id, input_keys=input_keys, dataset=dataset
    )
    actual_input_keys = _effective_input_keys(input_keys, inputs)

    # Resolve on-disk LR model dir (the model arg is unused when lr_model_id is given).
    model_dir = get_lr_model_dir(
        project,
        bug_id,
        model=config[0].model,
        dataset=dataset,
        lr_model_id=lr_model_id,
    )
    result_path = model_dir / LR_RESULT_FILE
    top5_path = model_dir / TOP5_FILE
    if result_path.exists() and top5_path.exists():
        logger.info(
            "Agent4LR %s-%s lr_model_id=%s: already complete, skipping",
            project,
            bug_id,
            lr_model_id,
        )
        return None

    # Refuse universes that don't follow FlexFL's exactly-20 combination rule.
    invalid_reason = candidate_list_skip_reason(inputs.candidates)
    if invalid_reason is not None:
        logger.info(
            "Agent4LR %s-%s lr_model_id=%s: %s, skipping",
            project,
            bug_id,
            lr_model_id,
            invalid_reason,
        )
        return None

    # Skip when there's no chance LR can recover the fault: either no
    # faulty methods are flagged, or none of them appear in the SR top-20.
    # Writing nothing on skip keeps the evaluation framework's "missing
    # = excluded from denominator" semantics correct (see plan).
    processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
    fault_sigs = _load_fault_signatures(processed_dir)
    if not _any_fault_in_candidates(inputs.candidates, fault_sigs):
        logger.info(
            "Agent4LR %s-%s lr_model_id=%s: no faulty method in top-20 "
            "(n_faults=%d, n_candidates=%d), skipping",
            project,
            bug_id,
            lr_model_id,
            len(fault_sigs),
            len(inputs.candidates),
        )
        return None

    candidate_source = f"FlexFL/SR/rankings/top20/{sr_model_id}.txt"
    inputs_desc = chain_inputs_descriptor(
        project=project,
        bug_id=bug_id,
        dataset=dataset,
        sr_model_id=sr_model_id,
        candidate_source=candidate_source,
        input_keys=tuple(actual_input_keys),
    )

    store = CheckpointStore(get_lr_checkpoints_dir(project, bug_id, dataset=dataset))
    full_chain: list[AgentSpec] = list(config)
    prefix_idx, ckpt = store.find_longest_prefix(inputs_desc, full_chain)
    logger.info(
        "Agent4LR %s-%s: prefix_idx=%d / %d",
        project,
        bug_id,
        prefix_idx,
        len(full_chain),
    )

    started_at = time.time()
    input_list: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    raw_responses: list[dict[str, Any]]
    if ckpt is not None:
        input_list = list(ckpt.input_list)
        messages = list(ckpt.messages)
        raw_responses = list(ckpt.raw_responses)
    else:
        seed_state = initial_state(inputs_desc=inputs_desc, inputs=inputs, chain=full_chain)
        input_list = seed_state.input_list
        messages = seed_state.messages
        raw_responses = seed_state.raw_responses

    ctx = LRToolContext(
        project=project, bug_id=bug_id_str, candidates=inputs.candidates, dataset=dataset
    )
    top5_indices: list[int] = []

    if prefix_idx == len(full_chain):
        # Full chain cached; pull top5 from the saved input_list.
        top5_indices = _extract_top5_from_cached_input_list(input_list)
    else:
        for k in range(prefix_idx, len(full_chain)):
            spec = full_chain[k]
            slot_hash = store.chain_hash(inputs_desc, full_chain[: k + 1])
            with store.acquire_slot(slot_hash, slot_index=k, agent_spec=spec) as lease:
                if lease.cached is not None:
                    # A peer process completed this slot first; adopt its state
                    # (full conversation up to and including slot k).
                    logger.info(
                        "Agent4LR %s-%s: slot %d (%s) adopted from peer checkpoint",
                        project,
                        bug_id,
                        k,
                        spec.role,
                    )
                    input_list = list(lease.cached.input_list)
                    messages = list(lease.cached.messages)
                    raw_responses = list(lease.cached.raw_responses)
                    if k == len(full_chain) - 1 and spec.role == "finisher":
                        top5_indices = _extract_top5_from_cached_input_list(input_list)
                    continue

                provider = _provider_for(
                    spec,
                    ollama_base_url=ollama_base_url,
                    ollama_verify=ollama_verify,
                    openai_api_key=openai_api_key,
                )
                if spec.role == "planner":
                    logger.info("Agent4LR %s-%s: planning (%s)", project, bug_id, spec.model)
                elif spec.role == "tool_caller":
                    logger.info(
                        "Agent4LR %s-%s: tool-loop (%s, up to %d iterations)",
                        project,
                        bug_id,
                        spec.model,
                        spec.iterations or 10,
                    )
                elif spec.role == "finisher":
                    logger.info("Agent4LR %s-%s: finisher (%s)", project, bug_id, spec.model)
                else:  # pragma: no cover — validate_v1_invariant rules this out
                    raise ValueError(f"Unknown slot role {spec.role!r}")

                # Drive the slot one API call at a time through the shared
                # stepper (the same functions the batch workflow uses).
                state = PendingSlotState(
                    inputs_descriptor=inputs_desc,
                    completed_chain=[s.identity() for s in full_chain[:k]],
                    slot_spec=spec.identity(),
                    calls_made=0,
                    exited=False,
                    input_list=input_list,
                    messages=messages,
                    raw_responses=raw_responses,
                )
                while not slot_is_complete(state):
                    req = build_request(state)
                    resp = provider.chat(
                        history=req.history, tools=req.tools, tool_choice=req.tool_choice
                    )
                    step = apply_response(state, resp, ctx)
                    state = step.state
                    if step.top5_indices is not None:
                        top5_indices = step.top5_indices
                input_list = list(state.input_list)
                messages = list(state.messages)
                raw_responses = list(state.raw_responses)

                lease.commit(
                    inputs_descriptor=inputs_desc,
                    completed_chain=full_chain[: k + 1],
                    input_list=input_list,
                    messages=messages,
                    raw_responses=raw_responses,
                )

    top5_fqns = rank_methods(ctx=ctx, top_5_methods=top5_indices)
    finished_at = time.time()

    result = LRRunResult(
        project=project,
        bug_id=bug_id_str,
        config_name=config_name,
        agent_chain=[s.identity() for s in full_chain],
        sr_model_id=sr_model_id,
        candidate_source=candidate_source,
        started_at=started_at,
        finished_at=finished_at,
        top5_indices=top5_indices,
        top5=top5_fqns,
        input_list=input_list,
        messages=messages,
        response_dumps=raw_responses,
        input=actual_input_keys,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    top5_path.write_text("\n".join(top5_fqns) + ("\n" if top5_fqns else ""), encoding="utf-8")
    logger.info(
        "Agent4LR %s-%s lr_model_id=%s: done in %.1fs, top5=%s",
        project,
        bug_id,
        lr_model_id,
        finished_at - started_at,
        top5_fqns,
    )
    return result


__all__ = [
    "DEFAULT_LR_INPUT_KEYS",
    "LR_RESULT_FILE",
    "TOP5_FILE",
    "LRConfig",
    "LRRunResult",
    "run_agent4lr_for_bug",
    "should_skip_no_fault",
]

# Silence "imported but unused" for the type alias re-export.
_ = CompletedCheckpoint
