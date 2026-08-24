"""Per-bug pipeline progress tracker.

Each processed bug directory holds a single ``tracker.json`` summarising:

* which extraction and FL pipeline steps have completed,
* per-step warnings and errors captured from the standard ``logging`` API,
* one entry per Agent4SR run (keyed by model_id), and
* coverage counts of faulty statements / methods / relevant tests / failing
  tests across the various rankings.

The tracker is **descriptive**, not prescriptive: it records what happened on
disk, but does not gate skip/re-run decisions. Each pipeline step retains its
existing ``skip_existing`` (file-existence) check; the tracker is updated
alongside whatever the step actually does.

Concurrency: writes are atomic (``os.replace``) but no inter-process locking
is performed. Callers are expected to run at most one process per bug at a
time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypedDict

from src.common.config import _SLUG_MAP, _model_slug, get_processed_dir

SCHEMA_VERSION = 2
TRACKER_FILENAME = "tracker.json"
DEFAULT_LOGGER_NAMES: tuple[str, ...] = ("src", "fl_methods")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class StepBucket(TypedDict):
    completed: list[str]
    warnings: dict[str, list[str]]
    errors: dict[str, list[str]]


class SRRunBucket(TypedDict):
    model: str
    temperature: float
    iterations: int
    base_url: str
    input: list[str]
    warnings: list[str]
    errors: list[str]


class LRRunBucket(TypedDict):
    """Tracker entry for one Agent4LR run.

    LR runs are **multi-agent**: a run is driven by a chain of
    ``AgentSpec``s (v1: planner → tool_caller → finisher). Per-slot
    parameters live inside ``agent_chain`` (a list of
    ``AgentSpec.identity()`` dicts, one per slot) rather than as
    bucket-level fields.

    ``config_name`` is the named-config slug (or ``"<adhoc>"`` for
    per-role CLI runs). ``candidate_source`` is the relative POSIX path
    to the top-20 file consumed (e.g.
    ``"FlexFL/SR/rankings/top20/llama3.1_8b.txt"``); ``sr_model_id``
    pins the LR run to a specific SR run for reproducibility.

    Identity tuple (see :func:`get_or_assign_lr_model_id`):
    ``(tuple(canonical_json(a) for a in agent_chain), sr_model_id,
    tuple(sorted(input)))``.
    """

    config_name: str
    agent_chain: list[dict[str, Any]]  # [AgentSpec.identity() per slot]
    sr_model_id: str
    candidate_source: str
    input: list[str]
    warnings: list[str]
    errors: list[str]


class CoverageEntry(TypedDict, total=False):
    count: int  # plus dynamic per-source keys (e.g., "ochiai", "boostn")


class Tracker(TypedDict):
    schema_version: int
    extraction: StepBucket
    fl: StepBucket
    sr: dict[str, SRRunBucket]
    lr: dict[str, LRRunBucket]
    coverage: dict[str, CoverageEntry]


Section = Literal["extraction", "fl"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _empty_step_bucket() -> StepBucket:
    return {"completed": [], "warnings": {}, "errors": {}}


def _empty_tracker() -> Tracker:
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction": _empty_step_bucket(),
        "fl": _empty_step_bucket(),
        "sr": {},
        "lr": {},
        "coverage": {},
    }


def tracker_path(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> Path:
    """Return the absolute path to ``tracker.json`` for one bug."""
    return get_processed_dir(project, bug_id, dataset=dataset) / TRACKER_FILENAME


def _migrate(data: dict[str, Any]) -> Tracker:
    """Schema migration shim.

    v1 → v2: introduces the ``lr`` bucket for Agent4LR runs. Migration
    is purely additive — existing v1 entries pass through unchanged with
    an empty ``lr`` dict added if absent.
    """
    version = data.get("schema_version", 1)
    if version < 2:
        data.setdefault("lr", {})
        data["schema_version"] = 2
    if "schema_version" not in data:
        data["schema_version"] = SCHEMA_VERSION
    return data  # type: ignore[return-value]


def load_tracker(
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
    create: bool = True,
) -> Tracker:
    """Load ``tracker.json`` for one bug.

    When the file is missing: returns an empty tracker if ``create=True``
    (default), else raises ``FileNotFoundError``.
    """
    path = tracker_path(project, bug_id, dataset=dataset)
    if not path.exists():
        if create:
            return _empty_tracker()
        raise FileNotFoundError(f"No tracker.json at {path}")
    return _migrate(json.loads(path.read_text(encoding="utf-8")))


def save_tracker(
    tracker: Tracker,
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
) -> None:
    """Atomically write ``tracker.json`` for one bug."""
    path = tracker_path(project, bug_id, dataset=dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def mark_completed(tracker: Tracker, section: Section, step: str) -> None:
    """Append ``step`` to ``tracker[section].completed`` if not already there."""
    completed = tracker[section]["completed"]
    if step not in completed:
        completed.append(step)


def record_warning(tracker: Tracker, section: Section, step: str, message: str) -> None:
    tracker[section]["warnings"].setdefault(step, []).append(message)


def record_error(tracker: Tracker, section: Section, step: str, message: str) -> None:
    tracker[section]["errors"].setdefault(step, []).append(message)


def record_sr_warning(tracker: Tracker, model_id: str, message: str) -> None:
    tracker["sr"][model_id]["warnings"].append(message)


def record_sr_error(tracker: Tracker, model_id: str, message: str) -> None:
    tracker["sr"][model_id]["errors"].append(message)


def record_lr_warning(tracker: Tracker, lr_model_id: str, message: str) -> None:
    tracker["lr"][lr_model_id]["warnings"].append(message)


def record_lr_error(tracker: Tracker, lr_model_id: str, message: str) -> None:
    tracker["lr"][lr_model_id]["errors"].append(message)


# ---------------------------------------------------------------------------
# SR model_id allocation
# ---------------------------------------------------------------------------


def _sr_identity(entry: SRRunBucket) -> tuple[str, float, int, tuple[str, ...]]:
    return (
        entry["model"],
        float(entry["temperature"]),
        int(entry["iterations"]),
        tuple(sorted(entry["input"])),
    )


def get_or_assign_sr_model_id(
    tracker: Tracker,
    *,
    model: str,
    temperature: float,
    iterations: int,
    base_url: str,
    input_keys: Sequence[str],
    slug_map: dict[str, str] = _SLUG_MAP,
) -> str:
    """Return the model_id to use for an Agent4SR run with the given config.

    Identity tuple: ``(model, temperature, iterations, sorted(input_keys))``.
    ``base_url`` is informational only — not part of identity.

    If any existing ``tracker.sr`` entry has a matching identity tuple, that
    entry's key is returned (allowing reuse of an existing run's outputs).
    Otherwise a new entry is allocated:

    * ``base = _model_slug(model, slug_map)`` is tried first;
    * if ``base`` is taken (by a different config), the first free
      ``f"{base}__{N}"`` (N >= 1) is used.

    Side effect: on allocation, an empty ``SRRunBucket`` carrying the
    request's config is inserted into ``tracker["sr"]``. The caller is
    responsible for persisting the tracker (typically immediately after).
    """
    target = (model, float(temperature), int(iterations), tuple(sorted(input_keys)))
    for mid, entry in tracker["sr"].items():
        if _sr_identity(entry) == target:
            return mid

    base = _model_slug(model, slug_map=slug_map)
    chosen = base
    if chosen in tracker["sr"]:
        n = 1
        while f"{base}__{n}" in tracker["sr"]:
            n += 1
        chosen = f"{base}__{n}"

    tracker["sr"][chosen] = {
        "model": model,
        "temperature": float(temperature),
        "iterations": int(iterations),
        "base_url": base_url,
        "input": list(input_keys),
        "warnings": [],
        "errors": [],
    }
    return chosen


# ---------------------------------------------------------------------------
# LR model_id allocation
# ---------------------------------------------------------------------------


def _canonical_agent_dict(a: dict[str, Any]) -> str:
    """Canonical-JSON serialisation of one ``AgentSpec.identity()`` dict."""
    return json.dumps(a, sort_keys=True, separators=(",", ":"))


def _lr_identity(
    entry: LRRunBucket,
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    return (
        tuple(_canonical_agent_dict(a) for a in entry["agent_chain"]),
        entry["sr_model_id"],
        tuple(sorted(entry["input"])),
    )


def _chain_hash_short(agent_chain: list[dict[str, Any]]) -> str:
    """First 12 hex chars of a stable hash of the agent chain.

    Used as the ``<adhoc>`` slug base so two adhoc invocations with
    identical chains land on the same ``lr_model_id`` without a tracker
    walk.
    """
    canonical = json.dumps(agent_chain, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()
    return digest[:12]


def find_lr_model_id(
    tracker: Tracker,
    *,
    agent_chain: list[dict[str, Any]],
    sr_model_id: str,
    input_keys: Sequence[str],
) -> str | None:
    """Return the existing ``lr_model_id`` for this identity, or ``None``.

    Read-only sibling of :func:`get_or_assign_lr_model_id` — same
    identity tuple, no bucket insertion. Used by the batch tick to check
    for existing results without mutating ``tracker.json``.
    """
    target = (
        tuple(_canonical_agent_dict(a) for a in agent_chain),
        sr_model_id,
        tuple(sorted(input_keys)),
    )
    for mid, entry in tracker["lr"].items():
        if _lr_identity(entry) == target:
            return mid
    return None


def get_or_assign_lr_model_id(
    tracker: Tracker,
    *,
    config_name: str,
    agent_chain: list[dict[str, Any]],
    sr_model_id: str,
    candidate_source: str,
    input_keys: Sequence[str],
    slug_map: dict[str, str] = _SLUG_MAP,
) -> str:
    """Return the ``lr_model_id`` slug to use for an Agent4LR run.

    Identity tuple: ``(tuple(canonical_json(a) for a in agent_chain),
    sr_model_id, tuple(sorted(input_keys)))``. ``candidate_source`` is
    informational — the SR run's ``sr_model_id`` is the canonical pin
    to a specific upstream candidate list.

    Slug allocation:

    * if ``config_name != "<adhoc>"``: base slug = ``_model_slug(config_name)``.
    * else: base slug = first 12 hex chars of the agent-chain hash
      (computed via :func:`_chain_hash_short`).
    * suffix ``__N`` (smallest free ``N`` ≥ 1) on collision when the
      base slug is taken by a *different* identity tuple.

    Side effect on allocation: an empty :class:`LRRunBucket` carrying
    the request's config is inserted into ``tracker["lr"]``. The caller
    is responsible for persisting the tracker (typically immediately
    after).
    """
    existing = find_lr_model_id(
        tracker,
        agent_chain=agent_chain,
        sr_model_id=sr_model_id,
        input_keys=input_keys,
    )
    if existing is not None:
        return existing

    if config_name != "<adhoc>":
        base = _model_slug(config_name, slug_map=slug_map)
    else:
        base = _chain_hash_short(agent_chain)
    chosen = base
    if chosen in tracker["lr"]:
        n = 1
        while f"{base}__{n}" in tracker["lr"]:
            n += 1
        chosen = f"{base}__{n}"

    tracker["lr"][chosen] = {
        "config_name": config_name,
        "agent_chain": list(agent_chain),
        "sr_model_id": sr_model_id,
        "candidate_source": candidate_source,
        "input": list(input_keys),
        "warnings": [],
        "errors": [],
    }
    return chosen


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def update_coverage(
    tracker: Tracker,
    *,
    s_faults: dict[str, int] | None = None,
    m_faults: dict[str, int] | None = None,
    r_tests: dict[str, int] | None = None,
    f_tests: dict[str, int] | None = None,
) -> None:
    """Replace coverage sub-entries. Pass ``None`` to leave a sub-entry unchanged."""
    if s_faults is not None:
        tracker["coverage"]["s_faults"] = dict(s_faults)  # type: ignore[assignment]
    if m_faults is not None:
        tracker["coverage"]["m_faults"] = dict(m_faults)  # type: ignore[assignment]
    if r_tests is not None:
        tracker["coverage"]["r_tests"] = dict(r_tests)  # type: ignore[assignment]
    if f_tests is not None:
        tracker["coverage"]["f_tests"] = dict(f_tests)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TrackerStep context manager
# ---------------------------------------------------------------------------


class _CapturingHandler(logging.Handler):
    """Buffers WARNING/ERROR log records into in-memory lists.

    Attached to specific named loggers by ``TrackerStep``; never to the root
    logger, to avoid capturing third-party library noise.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = record.msg if isinstance(record.msg, str) else str(record.msg)
        if record.levelno >= logging.ERROR:
            self.errors.append(msg)
        elif record.levelno >= logging.WARNING:
            self.warnings.append(msg)


class TrackerStep:
    """Context manager that records one pipeline step's outcome.

    Captures WARNING/ERROR records emitted by the configured loggers, plus
    any explicit ``record_warning``/``record_error`` calls, and flushes them
    into the appropriate tracker section on ``__exit__``. Marks the step
    ``completed`` on clean exit for extraction/fl sections.

    The context manager **never suppresses exceptions** and never gates
    execution. Callers retain their own ``skip_existing`` logic.

    Usage::

        with TrackerStep(project, bug_id, section="fl", step="boostn") as ts:
            try:
                run_boostn(...)
            except Exception as exc:
                # Existing handlers swallow exceptions; record explicitly so
                # the message survives into tracker.json.
                ts.record_error(repr(exc))
                ...

    For ``section="sr"``, ``model_id`` is required and the SR bucket should
    already exist (allocated via ``get_or_assign_sr_model_id`` and persisted
    before entering the context).
    """

    def __init__(
        self,
        project: str,
        bug_id: str | int,
        *,
        section: Literal["extraction", "fl", "sr", "lr"],
        step: str,
        model_id: str | None = None,
        dataset: str = "defects4j",
        logger_names: Sequence[str] | None = None,
        mark_complete_on_success: bool = True,
    ) -> None:
        if section in ("sr", "lr") and not model_id:
            raise ValueError(f"`model_id` is required when section={section!r}")
        self.project = project
        self.bug_id = bug_id
        self.section = section
        self.step = step
        self.model_id = model_id
        self.dataset = dataset
        self.logger_names: tuple[str, ...] = (
            tuple(logger_names) if logger_names is not None else DEFAULT_LOGGER_NAMES
        )
        self.mark_complete_on_success = mark_complete_on_success
        self._handler: _CapturingHandler | None = None
        self._attached: list[logging.Logger] = []
        self._explicit_warnings: list[str] = []
        self._explicit_errors: list[str] = []

    def __enter__(self) -> TrackerStep:
        self._handler = _CapturingHandler()
        for name in self.logger_names:
            lg = logging.getLogger(name)
            lg.addHandler(self._handler)
            self._attached.append(lg)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        # Detach handler first so subsequent code doesn't keep buffering.
        if self._handler is not None:
            for lg in self._attached:
                lg.removeHandler(self._handler)

        warnings = list(self._handler.warnings) if self._handler else []
        errors = list(self._handler.errors) if self._handler else []
        warnings.extend(self._explicit_warnings)
        errors.extend(self._explicit_errors)
        if exc is not None:
            errors.append(repr(exc))

        tracker = load_tracker(self.project, self.bug_id, dataset=self.dataset)

        if self.section == "sr":
            assert self.model_id is not None
            # The bucket should have been pre-allocated; create a placeholder
            # if not, so warnings/errors aren't dropped.
            if self.model_id not in tracker["sr"]:
                tracker["sr"][self.model_id] = {
                    "model": "",
                    "temperature": 0.0,
                    "iterations": 0,
                    "base_url": "",
                    "input": [],
                    "warnings": [],
                    "errors": [],
                }
            for w in warnings:
                record_sr_warning(tracker, self.model_id, w)
            for e in errors:
                record_sr_error(tracker, self.model_id, e)
        elif self.section == "lr":
            assert self.model_id is not None
            if self.model_id not in tracker["lr"]:
                tracker["lr"][self.model_id] = {
                    "config_name": "",
                    "agent_chain": [],
                    "sr_model_id": "",
                    "candidate_source": "",
                    "input": [],
                    "warnings": [],
                    "errors": [],
                }
            for w in warnings:
                record_lr_warning(tracker, self.model_id, w)
            for e in errors:
                record_lr_error(tracker, self.model_id, e)
        else:
            for w in warnings:
                record_warning(tracker, self.section, self.step, w)
            for e in errors:
                record_error(tracker, self.section, self.step, e)
            if exc is None and self.mark_complete_on_success:
                mark_completed(tracker, self.section, self.step)

        save_tracker(tracker, self.project, self.bug_id, dataset=self.dataset)
        return False

    def record_warning(self, message: str) -> None:
        """Explicit warning hook (flushed in ``__exit__``)."""
        self._explicit_warnings.append(message)

    def record_error(self, message: str) -> None:
        """Explicit error hook for callers whose ``except`` blocks swallow exceptions."""
        self._explicit_errors.append(message)
