"""Main Agent4SR agent loop: LLM interaction via Ollama ``/api/chat``.

Orchestrates the iterative tool-calling conversation:
  1. System prompt + initial user prompt (bug report + trigger test)
  2. Planning response from LLM
  3. Iterative tool calls (up to ``iterations`` rounds)
  4. Finisher prompt -> Top-5 culprit methods

Supports checkpoint-based resume: after each successful Ollama response
a checkpoint file is written so the pipeline can resume on restart.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from urllib3.exceptions import InsecureRequestWarning

from src.agent4sr.io import load_bug_inputs
from src.agent4sr.prompts import (
    sr_finisher_user_prompt,
    sr_initial_user_prompt,
    sr_retry_user_prompt,
    sr_system_prompt,
    sr_tool_call_user_prompt,
)
from src.agent4sr.tools import ToolContext, execute_tool, normalize_method_name, tool_schemas
from src.common.config import get_sr_model_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"
SR_RESULT_FILE = "sr_result.json"
TOP5_FILE = "top5.txt"
TOP5_RAW_FILE = "top5_raw.txt"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SRConfig:
    """Configuration for an Agent4SR run."""

    model: str
    iterations: int
    temperature: float
    base_url: str
    verify: bool
    bugs: tuple[str, ...]

    @staticmethod
    def from_cli(
        *,
        model: str,
        iterations: int,
        temperature: float,
        base_url: str,
        verify: bool,
        bugs: list[str] | tuple[str, ...],
    ) -> SRConfig:
        """Create config from CLI arguments."""
        norm_bugs = tuple(b.strip() for b in bugs if b.strip())
        return SRConfig(
            model=model,
            iterations=iterations,
            temperature=temperature,
            base_url=base_url,
            verify=verify,
            bugs=norm_bugs,
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


# Default static input list recorded in SRRunResult / sr_result.json. Mirrors
# the inputs assembled by ``load_bug_inputs``; surfaced explicitly so the
# tracker static-analyzer can reconstruct the SR run config from disk.
DEFAULT_SR_INPUT_KEYS: tuple[str, ...] = ("bug_report", "ochiai", "boostn", "sbir")


@dataclass(frozen=True)
class SRRunResult:
    """Result of a single Agent4SR run for one bug."""

    project: str
    bug_id: str
    model: str
    iterations: int
    temperature: float
    started_at: float
    finished_at: float
    final_content: str
    top5_raw: list[str]
    top5: list[str]
    transcript: list[dict[str, Any]]
    response_dumps: list[dict[str, Any]]
    base_url: str = ""
    input: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_top5(content: str) -> list[str]:
    """Extract Top_1 .. Top_5 method names from the finisher response."""
    found: dict[int, str] = {}
    for line in content.splitlines():
        s = line.strip()
        for i in range(1, 6):
            key = f"Top_{i}"
            if s.startswith(key) and ":" in s:
                rhs = s.split(":", 1)[1].strip()
                if rhs:
                    found[i] = rhs
    return [found[i] for i in range(1, 6) if i in found]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _is_complete(model_dir: Path) -> bool:
    """Check if all 3 output files exist (pipeline already finished)."""
    return (
        (model_dir / SR_RESULT_FILE).exists()
        and (model_dir / TOP5_FILE).exists()
        and (model_dir / TOP5_RAW_FILE).exists()
    )


def _save_checkpoint(
    model_dir: Path,
    *,
    messages: list[dict[str, Any]],
    raw_responses: list[dict[str, Any]],
    phase: str,
    iteration: int,
    started_at: float,
) -> None:
    """Atomically write checkpoint to disk."""
    data = {
        "messages": messages,
        "raw_responses": raw_responses,
        "phase": phase,
        "iteration": iteration,
        "started_at": started_at,
    }
    tmp_path = model_dir / "checkpoint.tmp.json"
    final_path = model_dir / CHECKPOINT_FILE
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)


def _load_checkpoint(model_dir: Path) -> dict[str, Any] | None:
    """Load checkpoint if it exists, else return None."""
    cp_path = model_dir / CHECKPOINT_FILE
    if not cp_path.exists():
        return None
    loaded: dict[str, Any] = json.loads(cp_path.read_text(encoding="utf-8"))
    return loaded


def _delete_checkpoint(model_dir: Path) -> None:
    """Remove checkpoint file after successful completion."""
    cp_path = model_dir / CHECKPOINT_FILE
    cp_path.unlink(missing_ok=True)
    # Also clean up tmp file if it exists
    (model_dir / "checkpoint.tmp.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent4sr_for_bug(
    *,
    project: str,
    bug_id: str | int,
    cfg: SRConfig,
    dataset: str = "defects4j",
    model_id: str | None = None,
) -> SRRunResult | None:
    """Run the full Agent4SR pipeline for a single bug.

    Parameters
    ----------
    project:
        Defects4J project identifier.
    bug_id:
        Bug/version number.
    cfg:
        Agent4SR configuration.
    model_id:
        Optional explicit on-disk dir name (lets the tracker assign a suffixed
        slug like ``llama3_1_8b__1`` so a second config gets its own
        ``Agent4SR/<model_id>/`` directory). Defaults to ``_model_slug(cfg.model)``.

    Returns
    -------
    SRRunResult | None
        Contains the Top-5 raw and normalised method names, transcript, etc.
        Returns None if the run was skipped (already complete).
    """
    bug_id_str = str(bug_id)
    model_dir = get_sr_model_dir(project, bug_id, cfg.model, dataset=dataset, model_id=model_id)

    # --- Skip if already complete ---
    if _is_complete(model_dir):
        logger.info("Agent4SR %s-%s %s: already complete, skipping.", project, bug_id, cfg.model)
        return None

    # --- Check for checkpoint (resume) ---
    checkpoint = _load_checkpoint(model_dir)

    raw_responses: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    resume_phase: str | None = None
    resume_iteration: int = 0

    if checkpoint is not None:
        messages = checkpoint["messages"]
        raw_responses = checkpoint["raw_responses"]
        resume_phase = checkpoint["phase"]
        resume_iteration = checkpoint["iteration"]
        t0 = checkpoint["started_at"]
        logger.info(
            "Agent4SR %s-%s %s: resuming from checkpoint (phase=%s, iteration=%d)",
            project,
            bug_id,
            cfg.model,
            resume_phase,
            resume_iteration,
        )
    else:
        inputs = load_bug_inputs(project, bug_id, dataset=dataset)
        raw_responses = []
        system = {
            "role": "system",
            "content": sr_system_prompt(max_tool_calls=cfg.iterations, dataset=dataset),
        }
        user0 = {"role": "user", "content": sr_initial_user_prompt(inputs)}
        messages = [system, user0]
        t0 = time.time()

    def _chat(
        *,
        msgs: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": cfg.temperature},
        }
        if tools is not None:
            payload["tools"] = tools
        if not cfg.verify:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InsecureRequestWarning)
                resp = requests.post(
                    f"{cfg.base_url}/api/chat",
                    json=payload,
                    timeout=300,
                    verify=False,
                )
        else:
            resp = requests.post(
                f"{cfg.base_url}/api/chat",
                json=payload,
                timeout=300,
                verify=True,
            )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return body

    # Step 1: Planning — LLM reasons about the bug (skip if resuming)
    if resume_phase is None:
        logger.info("Agent4SR %s-%s: planning...", project, bug_id)
        plan_resp = _chat(msgs=messages)
        raw_responses.append(plan_resp)
        messages.append(
            {
                "role": "assistant",
                "content": (plan_resp.get("message", {}) or {}).get("content") or "",
            }
        )
        _save_checkpoint(
            model_dir,
            messages=messages,
            raw_responses=raw_responses,
            phase="plan",
            iteration=0,
            started_at=t0,
        )

    # Step 2: Iterative tool calls
    ctx = ToolContext(project=project, bug_id=bug_id_str, dataset=dataset)
    schemas = tool_schemas(dataset)

    start_iteration = 0
    if resume_phase == "tools":
        start_iteration = resume_iteration + 1

    for iteration in range(start_iteration, cfg.iterations):
        messages.append({"role": "user", "content": sr_tool_call_user_prompt()})
        rd = _chat(msgs=messages, tools=schemas)
        raw_responses.append(rd)
        tool_calls = rd.get("message", {}).get("tool_calls")
        if not tool_calls:
            content = rd.get("message", {}).get("content") or ""
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": sr_retry_user_prompt()})
            _save_checkpoint(
                model_dir,
                messages=messages,
                raw_responses=raw_responses,
                phase="tools",
                iteration=iteration,
                started_at=t0,
            )
            continue

        tc0 = tool_calls[0]
        fn = (tc0.get("function") or {}).get("name")
        args = (tc0.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"method_name": args}
        if not isinstance(fn, str) or not isinstance(args, dict):
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content": sr_retry_user_prompt()})
            _save_checkpoint(
                model_dir,
                messages=messages,
                raw_responses=raw_responses,
                phase="tools",
                iteration=iteration,
                started_at=t0,
            )
            continue

        logger.debug(
            "Agent4SR %s-%s iter=%d: %s(%s)",
            project,
            bug_id,
            iteration,
            fn,
            json.dumps(args),
        )

        messages.append(
            {
                "role": "assistant",
                "content": rd.get("message", {}).get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        tool_out = execute_tool(ctx=ctx, name=fn, args=args)
        if fn == "exit":
            _save_checkpoint(
                model_dir,
                messages=messages,
                raw_responses=raw_responses,
                phase="tools",
                iteration=iteration,
                started_at=t0,
            )
            break
        messages.append({"role": "tool", "tool_name": fn, "content": tool_out})
        _save_checkpoint(
            model_dir,
            messages=messages,
            raw_responses=raw_responses,
            phase="tools",
            iteration=iteration,
            started_at=t0,
        )

    # Step 3: Finisher — ask for Top-5
    logger.info("Agent4SR %s-%s: requesting Top-5...", project, bug_id)
    messages.append({"role": "user", "content": sr_finisher_user_prompt(dataset=dataset)})
    finish_resp = _chat(msgs=messages)
    raw_responses.append(finish_resp)
    final_content = (finish_resp.get("message", {}) or {}).get("content") or ""
    messages.append({"role": "assistant", "content": final_content})

    # Parse and normalise Top-5
    top5_raw = _parse_top5(final_content)
    top5_norm: list[str] = []
    for m in top5_raw[:5]:
        top5_norm.append(normalize_method_name(ctx=ctx, method_name=m))

    t1 = time.time()
    logger.info(
        "Agent4SR %s-%s: done in %.1fs, top5=%s",
        project,
        bug_id,
        t1 - t0,
        top5_norm,
    )

    # --- Write final outputs ---
    result = SRRunResult(
        project=project,
        bug_id=bug_id_str,
        model=cfg.model,
        iterations=cfg.iterations,
        temperature=cfg.temperature,
        started_at=t0,
        finished_at=t1,
        final_content=final_content,
        top5_raw=top5_raw,
        top5=top5_norm,
        transcript=messages,
        response_dumps=raw_responses,
        base_url=cfg.base_url,
        input=list(DEFAULT_SR_INPUT_KEYS),
    )
    (model_dir / SR_RESULT_FILE).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    (model_dir / TOP5_FILE).write_text("\n".join(top5_norm) + "\n", encoding="utf-8")
    (model_dir / TOP5_RAW_FILE).write_text("\n".join(top5_raw) + "\n", encoding="utf-8")

    _delete_checkpoint(model_dir)
    return result
