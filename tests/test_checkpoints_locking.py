"""Tests for the slot-level lock coordination in ``src.agent4lr.checkpoints``.

Covers the new v2 API on :class:`CheckpointStore`:

* :meth:`acquire_slot` fast-path when ``<hash>.json`` already exists.
* :meth:`acquire_slot` lock acquisition when no peer holds the lock.
* Atomic ``commit`` writes the checkpoint and removes the lock.
* Context exit without ``commit`` releases the lock.
* Concurrent acquires: exactly one owner; others adopt the cached state.
* Stale-lock reclaim by dead PID, by TTL, and by corruption.
* Wait-timeout reclaim path.
* Lock-file format/JSON shape.

Most tests use ``threading.Thread`` (not multiprocessing) since the lock
protocol is keyed on file presence — the protocol is process-agnostic.
Each thread holds its own :class:`CheckpointStore` instance pointed at a
shared ``tmp_path`` so they exercise the same on-disk arena.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src.agent4lr.agents import AgentSpec
from src.agent4lr.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    LOCK_SCHEMA_VERSION,
    CheckpointStore,
    SlotLease,
)


@pytest.fixture
def planner_spec() -> AgentSpec:
    return AgentSpec(
        role="planner", provider="openai", model="gpt-5-mini", reasoning_effort="minimal"
    )


@pytest.fixture
def inputs_descriptor() -> dict[str, Any]:
    return {
        "project": "Lang",
        "bug_id": "1",
        "dataset": "defects4j",
        "sr_model_id": "llama3.1_8b",
        "candidate_source": "FlexFL/SR/rankings/top20/llama3.1_8b.txt",
        "input_keys": ["bug_report", "trigger_test"],
    }


def _slot_payload(
    inputs_descriptor: dict[str, Any],
    completed_chain: list[AgentSpec],
) -> dict[str, Any]:
    """Minimal valid commit payload for the tests below."""
    return {
        "inputs_descriptor": inputs_descriptor,
        "completed_chain": completed_chain,
        "input_list": [{"role": "user", "content": "stub"}],
        "messages": [{"role": "user", "content": "stub"}],
        "raw_responses": [{"id": "stub"}],
    }


# ---------------------------------------------------------------------------
# Fast path: existing checkpoint
# ---------------------------------------------------------------------------


def test_acquire_slot_fast_path_when_checkpoint_exists(
    tmp_path: Path, inputs_descriptor: dict[str, Any], planner_spec: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    # Pre-write a checkpoint at the slot's hash.
    store.save(**_slot_payload(inputs_descriptor, [planner_spec]))
    digest = store.chain_hash(inputs_descriptor, [planner_spec])

    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        assert lease.cached is not None, "expected cached state from existing checkpoint"
        assert lease.owned is False
        assert lease.cached.input_list == [{"role": "user", "content": "stub"}]
        assert lease.cached.schema_version == CHECKPOINT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Owning path
# ---------------------------------------------------------------------------


def test_acquire_slot_with_no_existing_returns_owned_lease(
    tmp_path: Path, planner_spec: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    digest = "a" * 32
    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        assert lease.cached is None
        assert lease.owned is True
        # Lock file should be present while we hold the lease.
        assert (tmp_path / f"{digest}.lock").exists()


def test_commit_writes_checkpoint_and_removes_lock(
    tmp_path: Path, inputs_descriptor: dict[str, Any], planner_spec: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    digest = store.chain_hash(inputs_descriptor, [planner_spec])
    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        assert lease.owned
        lease.commit(**_slot_payload(inputs_descriptor, [planner_spec]))

    assert (tmp_path / f"{digest}.json").exists(), "checkpoint must be on disk"
    assert not (tmp_path / f"{digest}.lock").exists(), "lock must be released on commit"
    # The lookup round-trip works
    ckpt = store.lookup(digest)
    assert ckpt is not None
    assert ckpt.input_list == [{"role": "user", "content": "stub"}]


def test_context_exit_without_commit_releases_lock(tmp_path: Path, planner_spec: AgentSpec) -> None:
    store = CheckpointStore(tmp_path)
    digest = "b" * 32

    class _Stop(Exception):
        pass

    with (
        pytest.raises(_Stop),
        store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease,
    ):
        assert lease.owned
        raise _Stop()

    assert not (tmp_path / f"{digest}.lock").exists(), "lock must be released on abort"
    assert not (tmp_path / f"{digest}.json").exists(), "no checkpoint should be written"


def test_commit_twice_raises(
    tmp_path: Path, inputs_descriptor: dict[str, Any], planner_spec: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    digest = store.chain_hash(inputs_descriptor, [planner_spec])
    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        lease.commit(**_slot_payload(inputs_descriptor, [planner_spec]))
        with pytest.raises(RuntimeError, match="twice"):
            lease.commit(**_slot_payload(inputs_descriptor, [planner_spec]))


def test_commit_on_non_owned_lease_raises(
    tmp_path: Path, inputs_descriptor: dict[str, Any], planner_spec: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    # Pre-write so the acquire returns a non-owning lease.
    store.save(**_slot_payload(inputs_descriptor, [planner_spec]))
    digest = store.chain_hash(inputs_descriptor, [planner_spec])
    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        assert not lease.owned
        with pytest.raises(RuntimeError, match="don't own"):
            lease.commit(**_slot_payload(inputs_descriptor, [planner_spec]))


# ---------------------------------------------------------------------------
# Lock file format
# ---------------------------------------------------------------------------


def test_lock_file_format(tmp_path: Path, planner_spec: AgentSpec) -> None:
    store = CheckpointStore(tmp_path)
    digest = "c" * 32
    with store.acquire_slot(digest, slot_index=2, agent_spec=planner_spec) as lease:
        assert lease.owned
        data = json.loads((tmp_path / f"{digest}.lock").read_text(encoding="utf-8"))

    assert data["schema_version"] == LOCK_SCHEMA_VERSION
    assert isinstance(data["pid"], int) and data["pid"] > 0
    assert data["hostname"] == socket.gethostname()
    assert isinstance(data["started_at"], (int, float))
    assert data["slot_index"] == 2
    assert data["agent_spec"]["role"] == "planner"
    assert data["agent_spec"]["model"] == "gpt-5-mini"


# ---------------------------------------------------------------------------
# Concurrency: two threads on the same hash
# ---------------------------------------------------------------------------


def test_concurrent_acquire_only_one_owns(
    tmp_path: Path, inputs_descriptor: dict[str, Any], planner_spec: AgentSpec
) -> None:
    """Two threads race for the same slot; one runs, the other adopts."""
    # IMPORTANT: the digest passed to acquire_slot must equal what commit()
    # recomputes internally from (inputs_descriptor, completed_chain), else
    # the waiter polls one hash while the holder writes to another.
    store_for_hash = CheckpointStore(tmp_path)
    digest = store_for_hash.chain_hash(inputs_descriptor, [planner_spec])
    payload = _slot_payload(inputs_descriptor, [planner_spec])
    results: dict[str, dict[str, Any]] = {}
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        store = CheckpointStore(tmp_path)
        barrier.wait()  # release both threads simultaneously
        with store.acquire_slot(
            digest,
            slot_index=0,
            agent_spec=planner_spec,
            poll_interval_sec=0.02,
            lock_ttl_sec=30,
            wait_timeout_sec=10,
        ) as lease:
            results[name] = {"owned": lease.owned, "cached": lease.cached is not None}
            if lease.owned:
                # Hold the lock briefly to ensure the other thread is forced to wait.
                time.sleep(0.2)
                lease.commit(**payload)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    owned = [name for name, r in results.items() if r["owned"]]
    cached = [name for name, r in results.items() if r["cached"]]
    assert len(owned) == 1, f"exactly one should own the lock, got {owned}"
    assert len(cached) == 1, f"the other should adopt cached state, got {cached}"
    assert (tmp_path / f"{digest}.json").exists()
    assert not (tmp_path / f"{digest}.lock").exists()


# ---------------------------------------------------------------------------
# Stale-lock reclaim
# ---------------------------------------------------------------------------


def _write_lock(
    path: Path,
    *,
    pid: int,
    hostname: str,
    started_at: float,
    slot_index: int = 0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "pid": pid,
                "hostname": hostname,
                "started_at": started_at,
                "slot_index": slot_index,
                "agent_spec": {"role": "planner", "model": "gpt-5-mini"},
            }
        ),
        encoding="utf-8",
    )


def test_stale_lock_dead_pid_reclaimed(tmp_path: Path, planner_spec: AgentSpec) -> None:
    digest = "e" * 32
    lock_path = tmp_path / f"{digest}.lock"
    # PID guaranteed not to exist (out of valid range on Linux).
    _write_lock(
        lock_path,
        pid=2**22 + 17,
        hostname=socket.gethostname(),
        started_at=time.time(),  # fresh
    )

    store = CheckpointStore(tmp_path)
    with store.acquire_slot(
        digest,
        slot_index=0,
        agent_spec=planner_spec,
        wait_timeout_sec=2.0,
        poll_interval_sec=0.05,
        lock_ttl_sec=120,
    ) as lease:
        assert lease.owned, "should reclaim a lock held by a dead PID"


def test_stale_lock_ttl_expired_reclaimed(tmp_path: Path, planner_spec: AgentSpec) -> None:
    digest = "f" * 32
    lock_path = tmp_path / f"{digest}.lock"
    # Use current PID (alive) but very old started_at to trigger TTL reclaim.
    _write_lock(
        lock_path,
        pid=99999999,  # also dead, but we want to test TTL specifically
        hostname="some-other-host",  # disable PID check
        started_at=time.time() - 99999,
    )

    store = CheckpointStore(tmp_path)
    with store.acquire_slot(
        digest,
        slot_index=0,
        agent_spec=planner_spec,
        wait_timeout_sec=2.0,
        poll_interval_sec=0.05,
        lock_ttl_sec=60,
    ) as lease:
        assert lease.owned, "should reclaim a TTL-expired lock"


def test_stale_lock_corrupt_reclaimed(tmp_path: Path, planner_spec: AgentSpec) -> None:
    digest = "0" * 32
    lock_path = tmp_path / f"{digest}.lock"
    lock_path.write_text("not valid json", encoding="utf-8")

    store = CheckpointStore(tmp_path)
    with store.acquire_slot(
        digest,
        slot_index=0,
        agent_spec=planner_spec,
        wait_timeout_sec=2.0,
        poll_interval_sec=0.05,
        lock_ttl_sec=60,
    ) as lease:
        assert lease.owned, "should reclaim a corrupt lock"


def test_wait_loop_times_out_and_reclaims(tmp_path: Path, planner_spec: AgentSpec) -> None:
    """If lock isn't stale and checkpoint never arrives, timeout reclaims."""
    digest = "1" * 32
    lock_path = tmp_path / f"{digest}.lock"
    # Lock owned by this very PID (so PID-liveness check returns alive) on a
    # fake hostname to disable same-host PID logic. Fresh timestamp.
    _write_lock(
        lock_path,
        pid=99999999,
        hostname="another-host",  # disables same-host PID check
        started_at=time.time(),
    )

    store = CheckpointStore(tmp_path)
    started = time.time()
    with store.acquire_slot(
        digest,
        slot_index=0,
        agent_spec=planner_spec,
        wait_timeout_sec=0.3,  # short wait
        poll_interval_sec=0.05,
        lock_ttl_sec=60,  # well above wait_timeout
    ) as lease:
        elapsed = time.time() - started
        assert lease.owned, "should reclaim after wait timeout"
        assert elapsed >= 0.25, "should actually wait roughly the configured timeout"


def test_max_retries_raises_on_pathological_contention(
    tmp_path: Path, planner_spec: AgentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the wait loop keeps reclaiming and another process keeps re-locking,
    eventually acquire_slot raises."""
    digest = "2" * 32
    store = CheckpointStore(tmp_path)
    lock_path = tmp_path / f"{digest}.lock"

    # Force the wait loop to always return "reclaim" by stubbing
    # _wait_for_checkpoint_or_reclaim to return None, and re-creating the lock
    # on every reclaim via _try_acquire_lock so EEXIST keeps firing.
    write_attempts = {"n": 0}

    def stubbed_try(p: Path, *, slot_index: int, agent_spec: dict[str, Any]) -> bool:
        # First call (after each reclaim) succeeds at "creating" the lock from
        # an external peer's perspective: we write the file, then return False
        # so the caller thinks we lost the race.
        write_attempts["n"] += 1
        if write_attempts["n"] >= 1:
            # Simulate another process re-creating the lock immediately.
            _write_lock(
                p,
                pid=99999999,
                hostname="another-host",
                started_at=time.time(),
            )
        return False

    monkeypatch.setattr(store, "_try_acquire_lock", stubbed_try)

    def fake_wait(**kwargs: Any) -> None:
        # Simulate reclaim every time.
        store._remove_lock(lock_path)
        return None

    monkeypatch.setattr(store, "_wait_for_checkpoint_or_reclaim", fake_wait)

    with pytest.raises(RuntimeError, match="pathological"):
        store.acquire_slot(
            digest,
            slot_index=0,
            agent_spec=planner_spec,
            wait_timeout_sec=0.1,
            poll_interval_sec=0.01,
            lock_ttl_sec=1,
        )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_slot_lease_repr_does_not_leak_internals(tmp_path: Path, planner_spec: AgentSpec) -> None:
    """Sanity check: SlotLease repr doesn't include the store or lock_path
    objects (just for log-readability)."""
    store = CheckpointStore(tmp_path)
    digest = "3" * 32
    with store.acquire_slot(digest, slot_index=0, agent_spec=planner_spec) as lease:
        r = repr(lease)
        assert "CheckpointStore" not in r
        assert isinstance(lease, SlotLease)
