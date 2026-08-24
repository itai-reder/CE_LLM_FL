"""Tests for ``PendingSlotState`` / ``PendingStateStore`` (batch mid-slot persistence).

Pins the contracts the batch workflow relies on:

* ``PendingSlotState.slot_hash()`` equals the hash the *completed*
  checkpoint for that slot will use (``chain_hash(inputs, chain +
  [slot_spec])``) — slot completion is a root-write + pending-delete.
* save/load round-trip, atomic write (no ``.tmp`` leftovers), delete
  idempotence, schema-version gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.checkpoints import (
    PENDING_SCHEMA_VERSION,
    CheckpointStore,
    PendingSlotState,
    PendingStateStore,
    chain_hash,
)

PLANNER = AgentSpec(role="planner", provider="openai", model="gpt-5-mini")
TOOL_CALLER = AgentSpec(role="tool_caller", provider="openai", model="gpt-5-nano", iterations=10)


@pytest.fixture
def inputs_desc() -> dict:
    return chain_inputs_descriptor(
        project="Lang",
        bug_id=1,
        dataset="defects4j",
        sr_model_id="llama3.1_8b",
        candidate_source="FlexFL/SR/rankings/top20/llama3.1_8b.txt",
        input_keys=("bug_report", "trigger_test"),
    )


@pytest.fixture
def state(inputs_desc: dict) -> PendingSlotState:
    return PendingSlotState(
        inputs_descriptor=inputs_desc,
        completed_chain=[PLANNER.identity()],
        slot_spec=TOOL_CALLER.identity(),
        calls_made=2,
        exited=False,
        input_list=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        raw_responses=[{"id": "resp_1"}],
    )


class TestSlotHash:
    def test_equals_completed_checkpoint_hash(
        self, tmp_path: Path, inputs_desc: dict, state: PendingSlotState
    ) -> None:
        store = CheckpointStore(tmp_path)
        expected = store.chain_hash(inputs_desc, [PLANNER, TOOL_CALLER])
        assert state.slot_hash() == expected

    def test_prefix_sharing_configs_share_hash(self, inputs_desc: dict) -> None:
        # Two configs identical through the tool_caller diverge only at
        # the finisher — their tool_caller pending states share a hash.
        a = chain_hash(inputs_desc, [PLANNER, TOOL_CALLER])
        b = chain_hash(inputs_desc, [PLANNER.identity(), TOOL_CALLER.identity()])
        assert a == b


class TestPendingStateStore:
    def test_save_load_round_trip(self, tmp_path: Path, state: PendingSlotState) -> None:
        store = PendingStateStore(tmp_path / "pending")
        path = store.save(state)
        assert path == store.path_for(state.slot_hash())
        loaded = store.load(state.slot_hash())
        assert loaded == state

    def test_no_tmp_leftovers(self, tmp_path: Path, state: PendingSlotState) -> None:
        store = PendingStateStore(tmp_path / "pending")
        store.save(state)
        assert list(store.root.glob("*.tmp.json")) == []

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        store = PendingStateStore(tmp_path / "pending")
        assert store.load("0" * 32) is None

    def test_schema_version_mismatch_is_miss(self, tmp_path: Path, state: PendingSlotState) -> None:
        store = PendingStateStore(tmp_path / "pending")
        path = store.save(state)
        content = path.read_text(encoding="utf-8").replace(
            f'"schema_version":{PENDING_SCHEMA_VERSION}', '"schema_version":0'
        )
        path.write_text(content, encoding="utf-8")
        assert store.load(state.slot_hash()) is None

    def test_delete_is_idempotent(self, tmp_path: Path, state: PendingSlotState) -> None:
        store = PendingStateStore(tmp_path / "pending")
        store.save(state)
        store.delete(state.slot_hash())
        assert store.load(state.slot_hash()) is None
        store.delete(state.slot_hash())  # second delete is a no-op

    def test_for_checkpoints_dir_roots_under_pending(self, tmp_path: Path) -> None:
        store = PendingStateStore.for_checkpoints_dir(tmp_path)
        assert store.root == tmp_path / "pending"

    def test_overwrite_advances_calls_made(self, tmp_path: Path, state: PendingSlotState) -> None:
        store = PendingStateStore(tmp_path / "pending")
        store.save(state)
        advanced = PendingSlotState(
            inputs_descriptor=state.inputs_descriptor,
            completed_chain=state.completed_chain,
            slot_spec=state.slot_spec,
            calls_made=3,
            exited=True,
            input_list=state.input_list,
            messages=state.messages,
            raw_responses=state.raw_responses,
        )
        assert advanced.slot_hash() == state.slot_hash()  # same identity, one file
        store.save(advanced)
        loaded = store.load(state.slot_hash())
        assert loaded is not None
        assert loaded.calls_made == 3
        assert loaded.exited is True
