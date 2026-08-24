"""The Agent4LR batch tick: process finished OpenAI batches, send the next round.

One tick advances every active (bug, config) agent by exactly one API
call, using the same stepper functions as the sync runner. The
high-level algorithm (repeatedly invoked via
``fl_methods/run_lr_batch.py``):

1. **Process** every unprocessed batch in the local registry: download
   outputs (including partial outputs of expired/cancelled batches),
   apply each line to its pending state — promoting completed slots to
   regular checkpoints — and mark the batch processed.
2. **Enumerate** (bug, config) pairs from the caller's selection.
3. **Filter/finalize**: skip pairs with existing results; pairs whose
   full chain is checkpointed are finalized through the injected
   ``finalize_fn`` (normally ``run_lr.run_lr_for_bug`` — a zero-LLM
   cache replay that writes ``lr_result.json``/``top5.txt``, tracker
   entries and top-5 rankings).
4. **Gate** bugs with insufficient inputs (BIP readiness, missing SR
   top-20, no fault in the candidate list).
5. **Load or create** the pending state for each pair's next slot.
6. **Merge** pairs sharing a slot hash — one request serves every
   config with the same chain prefix.
7. **Filter** states already in flight (custom_id present in any
   unprocessed batch, whatever its remote status).
8. **Build** request bodies via the shared
   :func:`~src.agent4lr.providers.openai.build_openai_request_body`.
9. **Send**: persist every included pending state (the pending file is
   the only custom_id → state binding), then upload one JSONL per model
   (Batch API allows a single model per input file) and register the
   created batches.

Failed/expired request lines leave their pending state untouched; the
next tick deterministically rebuilds and resends them. To abandon a
request permanently, delete its ``pending/<slot_hash>.json``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent4lr.agent import (
    LR_RESULT_FILE,
    TOP5_FILE,
    LRConfig,
    _effective_input_keys,
    candidate_list_skip_reason,
    should_skip_no_fault,
)
from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.batch_registry import BatchRecord, BatchRegistry, RequestKey
from src.agent4lr.checkpoints import (
    CheckpointStore,
    PendingSlotState,
    PendingStateStore,
    chain_hash,
)
from src.agent4lr.io import load_lr_bug_inputs
from src.agent4lr.providers.openai import build_openai_request_body, normalise_response_payload
from src.agent4lr.providers.openai_batch import (
    ACTIVE_BATCH_STATUSES,
    BATCH_ENDPOINT,
    BatchBackend,
)
from src.agent4lr.stepper import (
    apply_response,
    build_request,
    initial_state,
    slot_is_complete,
    state_for_slot,
)
from src.agent4lr.tools import LRToolContext
from src.common.bip_gate import bip_run_skip_reason
from src.common.config import FLEXFL_LR_SUBDIR, get_processed_dir
from src.common.tracker import find_lr_model_id, load_tracker

logger = logging.getLogger(__name__)

MAX_BATCH_BYTES = 180 * 1024 * 1024  # conservative margin under the 200 MB API cap

# finalize_fn(project=, bug_id=, config_name=, config=, sr_model_id=,
# input_keys=, dataset=) -> summary dict. Injected (normally
# run_lr.run_lr_for_bug) so this module stays independent of the
# top-level CLI scripts.
FinalizeFn = Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One deduped, ready-to-send batch request."""

    key: RequestKey
    state: PendingSlotState
    spec: AgentSpec
    body: dict[str, Any]
    config_names: tuple[str, ...]


@dataclass
class TickReport:
    """Everything one tick did (or would do, under dry_run)."""

    processed_batches: list[str] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)
    finalized: list[tuple[str, str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str, str, str]] = field(default_factory=list)
    in_flight: list[str] = field(default_factory=list)
    requests: list[WorkItem] = field(default_factory=list)
    batches_created: list[str] = field(default_factory=list)

    def skip(self, project: str, bug_id: str, config: str, reason: str) -> None:
        self.skipped.append((project, bug_id, config, reason))
        logger.info("Skipping %s-%s [%s]: %s", project, bug_id, config, reason)


# ---------------------------------------------------------------------------
# Path/store helpers (no mkdir side effects — safe under --dry-run)
# ---------------------------------------------------------------------------


def _checkpoints_dir_for(project: str, bug_id: str, dataset: str) -> Path:
    return get_processed_dir(project, bug_id, dataset=dataset) / FLEXFL_LR_SUBDIR / "checkpoints"


def _stores_for(
    project: str, bug_id: str, dataset: str
) -> tuple[CheckpointStore, PendingStateStore]:
    ckpt_dir = _checkpoints_dir_for(project, bug_id, dataset)
    return CheckpointStore(ckpt_dir), PendingStateStore.for_checkpoints_dir(ckpt_dir)


def _existing_result(
    project: str,
    bug_id: str,
    dataset: str,
    chain: LRConfig,
    sr_model_id: str,
    input_keys: Sequence[str],
) -> str | None:
    """Return the lr_model_id when results already exist on disk, else None."""
    tracker = load_tracker(project, bug_id, dataset=dataset)
    mid = find_lr_model_id(
        tracker,
        agent_chain=[s.identity() for s in chain],
        sr_model_id=sr_model_id,
        input_keys=list(input_keys),
    )
    if mid is None:
        return None
    model_dir = (
        get_processed_dir(project, bug_id, dataset=dataset) / FLEXFL_LR_SUBDIR / "Agent4LR" / mid
    )
    if (model_dir / LR_RESULT_FILE).exists() and (model_dir / TOP5_FILE).exists():
        return mid
    return None


def validate_openai_only(configs: dict[str, LRConfig]) -> None:
    """Hard-error on configs with non-OpenAI slots (batch is OpenAI-only)."""
    offenders = {
        name: [f"{s.role}={s.provider}" for s in chain if s.provider != "openai"]
        for name, chain in configs.items()
        if any(s.provider != "openai" for s in chain)
    }
    if offenders:
        raise ValueError(
            "The OpenAI Batch API workflow supports openai-provider slots only; "
            f"drop these configs or run them via run_lr.py: {offenders}"
        )


# ---------------------------------------------------------------------------
# Step 1 — process finished batches
# ---------------------------------------------------------------------------


def _iter_jsonl(content: bytes) -> Iterator[dict[str, Any]]:
    for line_no, raw in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line %d: %.200s", line_no, raw)


def apply_output_line(line: dict[str, Any]) -> str:
    """Apply one batch output line to its pending state.

    Returns an outcome tag (for reporting):

    * ``malformed`` — undecodable custom_id; dropped.
    * ``already_complete`` — the slot's completed checkpoint exists (a
      sync run won the race); response dropped, pending state removed.
    * ``missing_pending`` — no pending state on disk; dropped with a WARN.
    * ``stale`` — the state advanced past this request's call_index; dropped.
    * ``request_failed`` — error line / non-200 / incomplete response;
      pending state left untouched (the next tick resends).
    * ``applied`` — state advanced, slot still in progress.
    * ``slot_completed`` — state advanced and promoted to a completed
      checkpoint.
    """
    custom_id = str(line.get("custom_id") or "")
    try:
        key = RequestKey.decode(custom_id)
        dataset = key.dataset
    except ValueError as exc:
        logger.warning("Dropping output line with bad custom_id %r: %s", custom_id, exc)
        return "malformed"

    store, pstore = _stores_for(key.project, key.bug_id, dataset)
    if store.lookup(key.slot_hash) is not None:
        pstore.delete(key.slot_hash)
        return "already_complete"
    state = pstore.load(key.slot_hash)
    if state is None:
        logger.warning("No pending state for %s; dropping response", custom_id)
        return "missing_pending"
    if state.calls_made != key.call_index:
        logger.info(
            "Stale response %s (state at calls_made=%d); dropping", custom_id, state.calls_made
        )
        return "stale"

    error = line.get("error")
    response = line.get("response") or {}
    status_code = int(response.get("status_code") or 0)
    body = response.get("body") or {}
    body_status = body.get("status")
    if error or status_code != 200 or body_status not in (None, "completed"):
        logger.warning(
            "Request %s failed (status_code=%s, body_status=%s, error=%s); will resend next tick",
            custom_id,
            status_code,
            body_status,
            error,
        )
        return "request_failed"

    resp = normalise_response_payload(body)
    desc = state.inputs_descriptor
    inputs = load_lr_bug_inputs(
        key.project,
        key.bug_id,
        sr_model_id=str(desc["sr_model_id"]),
        input_keys=tuple(desc["input_keys"]),
        dataset=str(desc["dataset"]),
    )
    ctx = LRToolContext(
        project=key.project,
        bug_id=key.bug_id,
        candidates=inputs.candidates,
        dataset=str(desc["dataset"]),
    )
    step = apply_response(state, resp, ctx)
    if step.slot_completed:
        store.save(
            inputs_descriptor=step.state.inputs_descriptor,
            completed_chain=[*step.state.completed_chain, step.state.slot_spec],
            input_list=step.state.input_list,
            messages=step.state.messages,
            raw_responses=step.state.raw_responses,
        )
        pstore.delete(key.slot_hash)
        return "slot_completed"
    pstore.save(step.state)
    return "applied"


def process_finished_batches(
    *, registry: BatchRegistry, backend: BatchBackend
) -> tuple[list[str], dict[str, int]]:
    """Consume every terminal unprocessed batch; refresh statuses of active ones.

    Expired/cancelled batches are treated like completed ones — their
    partial output files are applied and requests missing from the
    output simply stay at their previous ``calls_made`` (resent by the
    caller's next send phase). Returns (processed batch ids, outcome
    counts).
    """
    processed: list[str] = []
    outcomes: dict[str, int] = {}
    for record in registry.unprocessed():
        remote = backend.retrieve_batch(record.batch_id)
        status = str(remote.get("status") or "")
        if status in ACTIVE_BATCH_STATUSES:
            if status != record.last_status:
                registry.mark(record.batch_id, last_status=status)
            continue

        output_file_id = remote.get("output_file_id")
        error_file_id = remote.get("error_file_id")
        if output_file_id:
            content = backend.download_file(str(output_file_id))
            registry.archive_path(record.batch_id, "output").write_bytes(content)
            for line in _iter_jsonl(content):
                outcome = apply_output_line(line)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if error_file_id:
            err_content = backend.download_file(str(error_file_id))
            registry.archive_path(record.batch_id, "errors").write_bytes(err_content)
            for line in _iter_jsonl(err_content):
                logger.error(
                    "Batch %s error line for %s: %s",
                    record.batch_id,
                    line.get("custom_id"),
                    line.get("error") or line.get("response"),
                )
        if status == "failed" and not output_file_id:
            logger.error(
                "Batch %s failed with no output file: %s", record.batch_id, remote.get("errors")
            )
        registry.mark(
            record.batch_id,
            last_status=status,
            processed=True,
            output_file_id=str(output_file_id) if output_file_id else None,
            error_file_id=str(error_file_id) if error_file_id else None,
        )
        processed.append(record.batch_id)
        logger.info("Processed batch %s (terminal status=%s)", record.batch_id, status)
    return processed, outcomes


# ---------------------------------------------------------------------------
# Steps 2-9 — the tick
# ---------------------------------------------------------------------------


@dataclass
class _PendingEntry:
    state: PendingSlotState
    config_names: set[str]


def _chunk_by_size(
    items: list[WorkItem], max_requests: int, max_bytes: int
) -> Iterator[list[WorkItem]]:
    chunk: list[WorkItem] = []
    chunk_bytes = 0
    for item in items:
        line_bytes = len(_request_line(item).encode("utf-8")) + 1
        if chunk and (len(chunk) >= max_requests or chunk_bytes + line_bytes > max_bytes):
            yield chunk
            chunk, chunk_bytes = [], 0
        chunk.append(item)
        chunk_bytes += line_bytes
    if chunk:
        yield chunk


def _request_line(item: WorkItem) -> str:
    return json.dumps(
        {
            "custom_id": item.key.encode(),
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": item.body,
        },
        separators=(",", ":"),
    )


def run_tick(
    *,
    dataset: str,
    bugs: Sequence[tuple[str, str]],
    configs: dict[str, LRConfig],
    sr_model_id: str,
    input_keys: tuple[str, ...],
    registry: BatchRegistry,
    backend: BatchBackend | None,
    finalize_fn: FinalizeFn | None = None,
    max_requests: int = 1000,
    max_batch_bytes: int = MAX_BATCH_BYTES,
    completion_window: str = "24h",
    dry_run: bool = False,
    no_send: bool = False,
) -> TickReport:
    """Run one idempotent batch tick (see module docstring for the steps).

    ``dry_run`` skips batch processing, finalization, and sending — the
    report shows what would happen. ``no_send`` processes and finalizes
    but sends nothing. ``backend`` may be ``None`` only when no API
    interaction can occur (dry_run, or no_send with nothing to process).
    """
    validate_openai_only(configs)
    report = TickReport()

    # Step 1 — process finished batches (a batch that completed the last
    # finisher call gets finalized by step 3 in this same invocation).
    if not dry_run and backend is not None:
        report.processed_batches, report.outcomes = process_finished_batches(
            registry=registry, backend=backend
        )

    # Steps 2-6 — enumerate, gate, filter, load/create, merge.
    pending_work: dict[tuple[str, str, str], _PendingEntry] = {}
    for project, bug_id in bugs:
        if dataset == "bugsinpy":
            reason = bip_run_skip_reason(project, bug_id)
            if reason is not None:
                report.skip(project, bug_id, "*", reason)
                continue
        try:
            inputs = load_lr_bug_inputs(
                project, bug_id, sr_model_id=sr_model_id, input_keys=input_keys, dataset=dataset
            )
        except FileNotFoundError:
            report.skip(project, bug_id, "*", f"SR top-20 missing (sr_model_id={sr_model_id})")
            continue
        invalid_reason = candidate_list_skip_reason(inputs.candidates)
        if invalid_reason is not None:
            report.skip(project, bug_id, "*", invalid_reason)
            continue
        if should_skip_no_fault(
            project=project, bug_id=bug_id, candidates=inputs.candidates, dataset=dataset
        ):
            report.skip(project, bug_id, "*", "no faulty method in the SR top-20")
            continue

        actual_keys = tuple(_effective_input_keys(input_keys, inputs))
        inputs_desc = chain_inputs_descriptor(
            project=project,
            bug_id=bug_id,
            dataset=dataset,
            sr_model_id=sr_model_id,
            candidate_source=f"FlexFL/SR/rankings/top20/{sr_model_id}.txt",
            input_keys=actual_keys,
        )
        store, pstore = _stores_for(project, bug_id, dataset)

        for config_name, chain in configs.items():
            chain_t = tuple(chain)
            existing = _existing_result(project, bug_id, dataset, chain_t, sr_model_id, actual_keys)
            if existing is not None:
                report.skip(project, bug_id, config_name, f"results exist ({existing})")
                continue

            prefix_idx, ckpt = store.find_longest_prefix(inputs_desc, chain_t)
            if prefix_idx == len(chain_t):
                # Step 3 — full chain checkpointed: zero-LLM finalization
                # through the sync path (results + tracker + rankings).
                if dry_run:
                    report.finalized.append((project, bug_id, config_name))
                    continue
                if finalize_fn is None:
                    report.skip(project, bug_id, config_name, "chain complete; no finalize_fn")
                    continue
                finalize_fn(
                    project=project,
                    bug_id=bug_id,
                    config_name=config_name,
                    config=chain_t,
                    sr_model_id=sr_model_id,
                    input_keys=input_keys,
                    dataset=dataset,
                )
                report.finalized.append((project, bug_id, config_name))
                continue

            slot_hash = chain_hash(inputs_desc, list(chain_t[: prefix_idx + 1]))
            work_key = (project, bug_id, slot_hash)
            entry = pending_work.get(work_key)
            if entry is not None:
                # Step 6 — another config shares this exact chain prefix.
                entry.config_names.add(config_name)
                continue

            state = pstore.load(slot_hash)
            if state is None:
                if ckpt is not None:
                    state = state_for_slot(
                        inputs_desc=inputs_desc, chain=chain_t, slot_index=prefix_idx, ckpt=ckpt
                    )
                else:
                    state = initial_state(inputs_desc=inputs_desc, inputs=inputs, chain=chain_t)
            if slot_is_complete(state):
                # Defensive: a complete pending state should have been
                # promoted at apply time. Promote now; the next tick
                # advances/finalizes from the checkpoint.
                if not dry_run:
                    store.save(
                        inputs_descriptor=state.inputs_descriptor,
                        completed_chain=[*state.completed_chain, state.slot_spec],
                        input_list=state.input_list,
                        messages=state.messages,
                        raw_responses=state.raw_responses,
                    )
                    pstore.delete(slot_hash)
                report.skip(
                    project, bug_id, config_name, "pending state was already complete; promoted"
                )
                continue
            pending_work[work_key] = _PendingEntry(state=state, config_names={config_name})

    # Step 7 — drop states already in flight; step 8 — build request bodies.
    in_flight = registry.in_flight_custom_ids()
    for (project, bug_id, slot_hash), entry in pending_work.items():
        key = RequestKey.for_state(
            dataset=dataset,
            project=project,
            bug_id=bug_id,
            slot_hash=slot_hash,
            call_index=entry.state.calls_made,
        )
        custom_id = key.encode()
        if custom_id in in_flight:
            report.in_flight.append(custom_id)
            continue
        req = build_request(entry.state)
        body = build_openai_request_body(
            model=req.spec.model,
            history=req.history,
            tools=req.tools,
            tool_choice=req.tool_choice,
            temperature=req.spec.temperature,
            top_p=req.spec.top_p,
            reasoning_effort=req.spec.reasoning_effort,
        )
        report.requests.append(
            WorkItem(
                key=key,
                state=entry.state,
                spec=req.spec,
                body=body,
                config_names=tuple(sorted(entry.config_names)),
            )
        )

    # Step 9 — persist pending states, upload, create batches.
    if dry_run or no_send or not report.requests:
        return report
    if backend is None:
        raise ValueError("backend is required to send batches (got None)")

    by_model: dict[str, list[WorkItem]] = {}
    for item in report.requests:
        by_model.setdefault(item.spec.model, []).append(item)

    for model, items in sorted(by_model.items()):
        for chunk in _chunk_by_size(items, max_requests, max_batch_bytes):
            # Pending files first — they are the only custom_id → state
            # binding, so they must exist before anything is in flight.
            for item in chunk:
                _, pstore = _stores_for(item.key.project, item.key.bug_id, dataset)
                pstore.save(item.state)
            content = ("\n".join(_request_line(item) for item in chunk) + "\n").encode("utf-8")
            input_file_id = backend.upload_jsonl(
                content, filename=f"agent4lr-{model}-{len(chunk)}reqs.jsonl"
            )
            batch = backend.create_batch(
                input_file_id=input_file_id,
                endpoint=BATCH_ENDPOINT,
                completion_window=completion_window,
                metadata={"cefl": "agent4lr", "benchmark": dataset, "model": model},
            )
            batch_id = str(batch["id"])
            registry.root.mkdir(parents=True, exist_ok=True)
            registry.archive_path(batch_id, "input").write_bytes(content)
            registry.add(
                BatchRecord(
                    batch_id=batch_id,
                    input_file_id=input_file_id,
                    endpoint=BATCH_ENDPOINT,
                    completion_window=completion_window,
                    model=model,
                    n_requests=len(chunk),
                    custom_ids=[item.key.encode() for item in chunk],
                    metadata={"cefl": "agent4lr", "benchmark": dataset},
                    last_status=str(batch.get("status") or "validating"),
                )
            )
            report.batches_created.append(batch_id)
            logger.info(
                "Created batch %s: %d requests, model=%s, window=%s",
                batch_id,
                len(chunk),
                model,
                completion_window,
            )
    return report


__all__ = [
    "MAX_BATCH_BYTES",
    "FinalizeFn",
    "TickReport",
    "WorkItem",
    "apply_output_line",
    "process_finished_batches",
    "run_tick",
    "validate_openai_only",
]
