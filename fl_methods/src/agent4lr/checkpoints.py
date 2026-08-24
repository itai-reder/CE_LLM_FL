"""Content-addressable checkpoint store for Agent4LR with slot-level locks.

The store enables **cross-config prefix reuse**: when two LR configs
share a leading sub-sequence of AgentSpecs, executing the second config
loads the longest matching prefix's checkpoint and only re-runs the
diverging suffix.

v2 adds **single-host slot-level locking** so parallel CLI invocations
targeting the same chain prefix do not double-execute that prefix. The
on-disk checkpoint format is unchanged from v1; locks are a sidecar
`<hash>.lock` file written next to the eventual `<hash>.json`. Lock
acquisition uses `O_CREAT|O_EXCL` for atomicity. Single-host POSIX
filesystems only; for distributed execution (e.g. across cluster
nodes) a real lock service would be needed.

Whole-slot granularity is preserved for *completed* checkpoints —
written once per completed slot, never mid-slot. Mid-slot interruption
of a sync run restarts that slot from scratch but the lock holder's
crash is detected by stale-lock reclaim (TTL or PID liveness on the
same host). The **batch workflow** additionally persists mid-slot
progress as :class:`PendingSlotState` files under ``pending/`` (see the
class docstring); the sync path neither reads nor writes those.

Layout::

    data/D4J/processed/<P>/<B>/FlexFL/LR/checkpoints/<hash>.json
    data/D4J/processed/<P>/<B>/FlexFL/LR/checkpoints/<hash>.lock  (transient)
    data/D4J/processed/<P>/<B>/FlexFL/LR/checkpoints/pending/<hash>.json  (batch only)

where ``<hash>`` = ``blake2b(canonical_json([inputs_descriptor] +
completed_chain), digest_size=16).hex()`` — 32 hex chars.

Each checkpoint file contains::

    {
      "schema_version": 2,
      "completed_chain": [AgentSpec.identity() per slot],
      "input_list": [...],
      "messages": [...],
      "raw_responses": [...],
      "finished_at": <unix-float>
    }

Each lock file contains::

    {
      "schema_version": 1,
      "pid": <int>,
      "hostname": <str>,
      "started_at": <unix-float>,
      "slot_index": <int>,
      "agent_spec": {AgentSpec.identity()}
    }

The runner reads ``input_list`` (canonical Responses-API-native
conversation state) to seed the LLM, and appends to both ``input_list``
and the human-readable ``messages`` view as further slots execute.
``raw_responses`` is preserved for auditability and parsing fallbacks.

Schema-v1 checkpoints (from before the input_list refactor) are not
loadable; the lookup treats them as misses and the slot reruns. Old
files remain on disk as orphans.

Usage pattern for the runner (see :mod:`src.agent4lr.agent`)::

    with store.acquire_slot(slot_hash, slot_index=k, agent_spec=spec) as lease:
        if lease.cached is not None:
            # Another process completed this slot; adopt its state.
            input_list = list(lease.cached.input_list)
            ...
            continue
        # Lock owned; do LLM work, then commit.
        ...
        lease.commit(
            inputs_descriptor=desc,
            completed_chain=full_chain[: k + 1],
            input_list=input_list, messages=messages, raw_responses=raw,
        )

Direct :meth:`CheckpointStore.save` calls remain supported (used by
:meth:`SlotLease.commit` internally and by tests that don't exercise
the lock path) but the lease-based API is the recommended path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from src.agent4lr.agents import AgentSpec

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 2
LOCK_SCHEMA_VERSION = 1


def _as_identity(slot: dict[str, Any] | AgentSpec) -> dict[str, Any]:
    return slot.identity() if isinstance(slot, AgentSpec) else dict(slot)


def chain_hash(
    inputs_descriptor: dict[str, Any],
    completed_chain: Sequence[dict[str, Any] | AgentSpec],
) -> str:
    """Compute the deterministic hash key for a chain prefix.

    ``completed_chain`` may be a sequence of ``AgentSpec`` instances or
    pre-computed identity dicts; both produce identical hashes for
    matching content. Output is 32 hex chars (blake2b-128). Module-level
    so batch tooling can key pending states without a store instance;
    :meth:`CheckpointStore.chain_hash` delegates here.
    """
    normalised = [_as_identity(c) for c in completed_chain]
    payload = json.dumps(
        {"inputs": inputs_descriptor, "chain": normalised},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


DEFAULT_WAIT_TIMEOUT_SEC = 1800.0  # 30 min — caller should rarely hit this
DEFAULT_POLL_INTERVAL_SEC = 2.0  # checkpoint-file poll cadence in waiters
DEFAULT_LOCK_TTL_SEC = 900.0  # 15 min — comfortably > slowest observed slot (M2R4 ~217s)
_MAX_ACQUIRE_RETRIES = 10  # bound the acquire→wait→reclaim→retry loop


@dataclass(frozen=True)
class CompletedCheckpoint:
    """One completed-slot checkpoint as loaded from disk."""

    completed_chain: list[dict[str, Any]]
    input_list: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    raw_responses: list[dict[str, Any]]
    finished_at: float
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


@dataclass
class SlotLease:
    """A lease on a slot's computation, returned by :meth:`CheckpointStore.acquire_slot`.

    Context manager. Two terminal states:

    * ``cached is not None`` and ``owned`` is False → another process
      already completed this slot. Caller adopts ``cached`` and does no
      work. Lock (if any) is held by that other process; this lease
      holds nothing.
    * ``cached is None`` and ``owned`` is True → caller holds the lock
      and must either call :meth:`commit` after running the slot, or
      let the context manager exit without commit (which removes the
      lock so other waiters can retry).
    """

    cached: CompletedCheckpoint | None
    owned: bool
    _store: CheckpointStore = field(repr=False)
    _hash: str = ""
    _lock_path: Path | None = field(default=None, repr=False)
    _committed: bool = field(default=False, repr=False)

    def __enter__(self) -> SlotLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # If we held the lock but never committed (e.g., exception during slot
        # execution, or test/dev abort), drop the lock so the next process can
        # retry the slot from scratch.
        if self.owned and not self._committed and self._lock_path is not None:
            self._store._remove_lock(self._lock_path)

    def commit(
        self,
        *,
        inputs_descriptor: dict[str, Any],
        completed_chain: Sequence[dict[str, Any] | AgentSpec],
        input_list: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        raw_responses: list[dict[str, Any]],
    ) -> Path:
        """Write the checkpoint atomically, then remove the lock.

        Only meaningful when ``owned`` is True. ``commit()`` is idempotent
        per-lease — calling it twice is a runtime error.
        """
        if not self.owned:
            raise RuntimeError(
                "Cannot commit a SlotLease we don't own (cached state was adopted instead)."
            )
        if self._committed:
            raise RuntimeError("SlotLease.commit() called twice.")
        path = self._store.save(
            inputs_descriptor=inputs_descriptor,
            completed_chain=completed_chain,
            input_list=input_list,
            messages=messages,
            raw_responses=raw_responses,
        )
        self._committed = True
        if self._lock_path is not None:
            self._store._remove_lock(self._lock_path)
        return path


class CheckpointStore:
    """Per-bug checkpoint directory.

    Instances are cheap; one is constructed per bug at the start of a
    run. All IO is bounded to ``root``. Atomic writes via tmp + os.replace.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the store at ``root`` (created lazily on save).

        ``root`` is usually
        ``data/D4J/processed/<P>/<B>/FlexFL/LR/checkpoints/`` — see
        :func:`src.common.config.get_lr_checkpoints_dir`.
        """
        self.root = root

    @staticmethod
    def _as_identity(slot: dict[str, Any] | AgentSpec) -> dict[str, Any]:
        return _as_identity(slot)

    def chain_hash(
        self,
        inputs_descriptor: dict[str, Any],
        completed_chain: Sequence[dict[str, Any] | AgentSpec],
    ) -> str:
        """Compute the deterministic hash key for a chain prefix.

        Delegates to the module-level :func:`chain_hash`; kept as a
        method for existing call sites.
        """
        return chain_hash(inputs_descriptor, completed_chain)

    def lookup(self, hash_str: str) -> CompletedCheckpoint | None:
        """Return the checkpoint stored under ``hash_str``, or ``None``.

        Reads ``<root>/<hash_str>.json``. Missing file → ``None``.
        Files with a non-current ``schema_version`` are treated as
        cache misses (logged at DEBUG) so old checkpoints from before
        the input_list refactor stay on disk harmlessly while new runs
        regenerate them. Malformed JSON still raises — that's a real
        corruption signal.
        """
        path = self.root / f"{hash_str}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version", 0)
        if version != CHECKPOINT_SCHEMA_VERSION:
            logger.debug(
                "Skipping checkpoint %s: schema_version=%r != %d",
                path,
                version,
                CHECKPOINT_SCHEMA_VERSION,
            )
            return None
        return CompletedCheckpoint(
            completed_chain=data["completed_chain"],
            input_list=data["input_list"],
            messages=data["messages"],
            raw_responses=data["raw_responses"],
            finished_at=float(data["finished_at"]),
            schema_version=int(version),
        )

    def find_longest_prefix(
        self,
        inputs_descriptor: dict[str, Any],
        full_chain: Sequence[AgentSpec],
    ) -> tuple[int, CompletedCheckpoint | None]:
        """Return the longest cached prefix of ``full_chain``.

        Walks ``range(len(full_chain), -1, -1)`` and returns the first
        prefix length ``k`` for which a checkpoint exists. ``k == 0``
        with ``checkpoint=None`` means no cached prefix. ``k ==
        len(full_chain)`` means the entire chain is already computed
        (final outputs can be written from the loaded checkpoint
        without further LLM calls).
        """
        for k in range(len(full_chain), -1, -1):
            digest = self.chain_hash(inputs_descriptor, full_chain[:k])
            ckpt = self.lookup(digest)
            if ckpt is not None:
                return k, ckpt
        return 0, None

    def save(
        self,
        *,
        inputs_descriptor: dict[str, Any],
        completed_chain: Sequence[dict[str, Any] | AgentSpec],
        input_list: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        raw_responses: list[dict[str, Any]],
    ) -> Path:
        """Atomically write a completed-slot checkpoint to disk.

        Writes to ``<root>/<hash>.tmp.json`` then ``os.replace`` to
        ``<hash>.json``. Returns the final path. ``finished_at`` is set
        from ``time.time()`` at call time.

        Direct callers bypass the lock protocol — that's intentional for
        tests and the lease's own commit path. Runner code should use
        :meth:`acquire_slot` instead.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        digest = self.chain_hash(inputs_descriptor, completed_chain)
        final_path = self.root / f"{digest}.json"
        tmp_path = self.root / f"{digest}.tmp.json"
        normalised = [self._as_identity(c) for c in completed_chain]
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "completed_chain": normalised,
            "input_list": input_list,
            "messages": messages,
            "raw_responses": raw_responses,
            "finished_at": time.time(),
        }
        tmp_path.write_text(
            json.dumps(payload, sort_keys=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, final_path)
        logger.debug("Wrote checkpoint %s (%d slots)", digest, len(normalised))
        return final_path

    # ------------------------------------------------------------------
    # Slot-level locking (v2)
    # ------------------------------------------------------------------

    def acquire_slot(
        self,
        hash_str: str,
        *,
        slot_index: int,
        agent_spec: AgentSpec | dict[str, Any],
        wait_timeout_sec: float = DEFAULT_WAIT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        lock_ttl_sec: float = DEFAULT_LOCK_TTL_SEC,
    ) -> SlotLease:
        """Reserve a slot for computation or wait for a peer's result.

        Returns a :class:`SlotLease`. Always use as a context manager so
        the lock is released on exception::

            with store.acquire_slot(digest, slot_index=k, agent_spec=spec) as lease:
                if lease.cached is not None:
                    ...  # adopt cached state, skip LLM work
                else:
                    ...  # run the slot
                    lease.commit(...)

        Coordination protocol:

        1. Fast path — if ``<hash>.json`` already exists, return a
           non-owning lease with ``cached`` populated.
        2. Atomic lock attempt via ``O_CREAT|O_EXCL``. On success,
           return an owning lease.
        3. On EEXIST, wait (polling ``<hash>.json``) until either the
           checkpoint appears (return non-owning lease) or the lock is
           detected as stale (TTL exceeded, or PID dead on same host).
           Stale locks are reclaimed and the protocol loops.
        4. After ``_MAX_ACQUIRE_RETRIES`` rounds without progress, raise
           ``RuntimeError`` — this should not happen in practice and
           indicates either a bug or pathological contention.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f"{hash_str}.lock"
        spec_identity = self._as_identity(agent_spec)

        for attempt in range(_MAX_ACQUIRE_RETRIES):
            # Fast path: checkpoint already exists.
            ckpt = self.lookup(hash_str)
            if ckpt is not None:
                return SlotLease(
                    cached=ckpt,
                    owned=False,
                    _store=self,
                    _hash=hash_str,
                    _lock_path=None,
                )

            # Try to atomically create the lock file.
            if self._try_acquire_lock(lock_path, slot_index=slot_index, agent_spec=spec_identity):
                return SlotLease(
                    cached=None,
                    owned=True,
                    _store=self,
                    _hash=hash_str,
                    _lock_path=lock_path,
                )

            # Someone else holds the lock; wait for the checkpoint or reclaim.
            result = self._wait_for_checkpoint_or_reclaim(
                hash_str=hash_str,
                lock_path=lock_path,
                wait_timeout_sec=wait_timeout_sec,
                poll_interval_sec=poll_interval_sec,
                lock_ttl_sec=lock_ttl_sec,
            )
            if isinstance(result, CompletedCheckpoint):
                return SlotLease(
                    cached=result,
                    owned=False,
                    _store=self,
                    _hash=hash_str,
                    _lock_path=None,
                )
            # else: result is None → lock was reclaimed; loop and retry acquire.
            logger.debug(
                "acquire_slot(%s): retry %d/%d after stale-lock reclaim",
                hash_str,
                attempt + 1,
                _MAX_ACQUIRE_RETRIES,
            )

        raise RuntimeError(
            f"acquire_slot({hash_str}) gave up after {_MAX_ACQUIRE_RETRIES} retries; "
            "this indicates pathological contention or a bug in the lock protocol."
        )

    def _lock_payload(
        self,
        *,
        slot_index: int,
        agent_spec: dict[str, Any],
    ) -> bytes:
        payload: dict[str, Any] = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "slot_index": slot_index,
            "agent_spec": agent_spec,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _try_acquire_lock(
        self,
        lock_path: Path,
        *,
        slot_index: int,
        agent_spec: dict[str, Any],
    ) -> bool:
        """Attempt atomic O_EXCL creation. Returns True on success."""
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        try:
            os.write(fd, self._lock_payload(slot_index=slot_index, agent_spec=agent_spec))
        finally:
            os.close(fd)
        return True

    def _remove_lock(self, lock_path: Path) -> None:
        """Remove the lock file; ignore if already gone."""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove lock %s: %s", lock_path, exc)

    def _read_lock(self, lock_path: Path) -> dict[str, Any] | None:
        """Read and parse the lock file. Returns None if missing/unparseable."""
        try:
            raw = lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Failed to read lock %s: %s", lock_path, exc)
            return None
        if not raw.strip():
            # Lock holder created the file but crashed before writing payload.
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt lock file %s; treating as stale.", lock_path)
            return {}
        if isinstance(data, dict):
            return data
        return {}

    def _is_stale_lock(self, lock_path: Path, lock_ttl_sec: float) -> bool:
        """Return True if the lock at ``lock_path`` should be reclaimed."""
        data = self._read_lock(lock_path)
        if data is None:
            # Lock disappeared between checks; treat as gone (not stale, but
            # the caller will discover via re-attempted acquire).
            return False
        if not data:
            # Empty or unparseable — stale.
            return True
        started_at = data.get("started_at")
        if not isinstance(started_at, (int, float)):
            return True
        if time.time() - float(started_at) > lock_ttl_sec:
            return True
        # Same-host PID liveness check
        lock_host = data.get("hostname")
        lock_pid = data.get("pid")
        return (
            isinstance(lock_host, str)
            and isinstance(lock_pid, int)
            and lock_host == socket.gethostname()
            and not _pid_alive(lock_pid)
        )

    def _wait_for_checkpoint_or_reclaim(
        self,
        *,
        hash_str: str,
        lock_path: Path,
        wait_timeout_sec: float,
        poll_interval_sec: float,
        lock_ttl_sec: float,
    ) -> CompletedCheckpoint | None:
        """Block until the checkpoint appears or we reclaim a stale lock.

        Returns the checkpoint (on success) or ``None`` (we reclaimed the
        lock; caller should retry acquire).
        """
        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            ckpt = self.lookup(hash_str)
            if ckpt is not None:
                return ckpt
            if self._is_stale_lock(lock_path, lock_ttl_sec):
                self._remove_lock(lock_path)
                return None
            time.sleep(poll_interval_sec)
        logger.warning(
            "Wait timeout (%.0fs) for checkpoint %s; forcing lock reclaim.",
            wait_timeout_sec,
            hash_str,
        )
        self._remove_lock(lock_path)
        return None


def _pid_alive(pid: int) -> bool:
    """Return True iff the given PID is alive on this host.

    Uses ``os.kill(pid, 0)``: success or ``PermissionError`` (PID exists,
    different owner) means alive; ``ProcessLookupError`` means dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # On Linux, kill(pid, 0) shouldn't raise anything else, but be safe:
        # ESRCH = no such process; anything else = treat as alive.
        return exc.errno != errno.ESRCH
    return True


# ---------------------------------------------------------------------------
# Pending (mid-slot) states — batch-API support
# ---------------------------------------------------------------------------

PENDING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PendingSlotState:
    """Mid-slot progress for the batch workflow.

    Represents the conversation state after ``calls_made`` *complete*
    request→response→tool-execution rounds inside the slot described by
    ``slot_spec`` (an ``AgentSpec.identity()`` dict). Ephemeral pre-call
    messages (the tool-loop reminder, the finisher handoff) are **not**
    persisted dangling — the stepper regenerates them deterministically
    when building the next request and appends them together with the
    response items, so resending a lost request is idempotent.

    ``slot_hash()`` equals the hash the *completed* checkpoint for this
    slot will use, so slot completion is literally: write
    ``<slot_hash>.json`` at the store root, delete
    ``pending/<slot_hash>.json``. Two configs sharing a chain prefix
    therefore share one pending state (and one batch request) until
    their specs diverge — the same guarantee as sync prefix reuse.
    """

    inputs_descriptor: dict[str, Any]
    completed_chain: list[dict[str, Any]]
    slot_spec: dict[str, Any]
    calls_made: int
    exited: bool
    input_list: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    raw_responses: list[dict[str, Any]]
    schema_version: int = PENDING_SCHEMA_VERSION

    def slot_hash(self) -> str:
        """Hash of the slot this state is progressing towards completing."""
        return chain_hash(self.inputs_descriptor, [*self.completed_chain, self.slot_spec])


class PendingStateStore:
    """Persistence for :class:`PendingSlotState` under ``<checkpoints_dir>/pending/``.

    Same atomicity discipline as :class:`CheckpointStore` (tmp +
    ``os.replace``); no locking — pending states are only written by the
    single-instance batch tick, and the completed-checkpoint store
    remains the authority on finished slots.
    """

    SUBDIR = "pending"

    def __init__(self, root: Path) -> None:
        """``root`` is the *pending* directory itself, created lazily on save."""
        self.root = root

    @classmethod
    def for_checkpoints_dir(cls, checkpoints_dir: Path) -> PendingStateStore:
        """Construct rooted at ``<checkpoints_dir>/pending/``."""
        return cls(checkpoints_dir / cls.SUBDIR)

    def path_for(self, slot_hash: str) -> Path:
        return self.root / f"{slot_hash}.json"

    def load(self, slot_hash: str) -> PendingSlotState | None:
        """Return the pending state stored under ``slot_hash``, or ``None``.

        Missing file → ``None``. Non-current ``schema_version`` → treated
        as a miss (logged at DEBUG) so the slot restarts from the last
        completed checkpoint. Malformed JSON still raises — that's a real
        corruption signal.
        """
        path = self.path_for(slot_hash)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version", 0)
        if version != PENDING_SCHEMA_VERSION:
            logger.debug(
                "Skipping pending state %s: schema_version=%r != %d",
                path,
                version,
                PENDING_SCHEMA_VERSION,
            )
            return None
        return PendingSlotState(
            inputs_descriptor=data["inputs_descriptor"],
            completed_chain=data["completed_chain"],
            slot_spec=data["slot_spec"],
            calls_made=int(data["calls_made"]),
            exited=bool(data["exited"]),
            input_list=data["input_list"],
            messages=data["messages"],
            raw_responses=data["raw_responses"],
            schema_version=int(version),
        )

    def save(self, state: PendingSlotState) -> Path:
        """Atomically write ``state`` to ``pending/<slot_hash>.json``."""
        self.root.mkdir(parents=True, exist_ok=True)
        digest = state.slot_hash()
        final_path = self.path_for(digest)
        tmp_path = self.root / f"{digest}.tmp.json"
        payload: dict[str, Any] = {
            "schema_version": PENDING_SCHEMA_VERSION,
            "slot_hash": digest,
            "inputs_descriptor": state.inputs_descriptor,
            "completed_chain": state.completed_chain,
            "slot_spec": state.slot_spec,
            "calls_made": state.calls_made,
            "exited": state.exited,
            "input_list": state.input_list,
            "messages": state.messages,
            "raw_responses": state.raw_responses,
            "updated_at": time.time(),
        }
        tmp_path.write_text(
            json.dumps(payload, sort_keys=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, final_path)
        logger.debug("Wrote pending state %s (calls_made=%d)", digest, state.calls_made)
        return final_path

    def delete(self, slot_hash: str) -> None:
        """Remove the pending state; ignore if already gone."""
        self.path_for(slot_hash).unlink(missing_ok=True)
