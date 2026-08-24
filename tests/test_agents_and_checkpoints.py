"""Unit tests for ``src.agent4lr.agents`` and ``src.agent4lr.checkpoints``.

Covers:

* ``AgentSpec.identity()`` / ``identity_canonical_json()`` determinism.
* ``chain_inputs_descriptor`` sorting and shape.
* ``CheckpointStore.chain_hash`` determinism and sensitivity.
* ``CheckpointStore.save`` / ``lookup`` round-trip with atomic writes.
* ``CheckpointStore.find_longest_prefix`` returning the longest cached
  prefix; ``(0, None)`` on miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
    CompletedCheckpoint,
)


@pytest.fixture
def planner() -> AgentSpec:
    return AgentSpec(
        role="planner", provider="openai", model="gpt-4.1-mini", reasoning_effort="low"
    )


@pytest.fixture
def tool_caller() -> AgentSpec:
    return AgentSpec(
        role="tool_caller",
        provider="ollama",
        model="llama3.1:8b",
        iterations=10,
    )


@pytest.fixture
def finisher() -> AgentSpec:
    return AgentSpec(role="finisher", provider="openai", model="gpt-4.1-mini")


@pytest.fixture
def inputs() -> dict:
    return chain_inputs_descriptor(
        project="Lang",
        bug_id="1",
        dataset="defects4j",
        sr_model_id="llama3.1_8b",
        candidate_source="FlexFL/SR/rankings/top20/llama3.1_8b.txt",
        input_keys=("bug_report", "trigger_test"),
    )


# ---------------------------------------------------------------------------
# AgentSpec
# ---------------------------------------------------------------------------


def test_agent_spec_identity_is_a_dict_with_all_fields(planner: AgentSpec) -> None:
    ident = planner.identity()
    assert ident == {
        "role": "planner",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_effort": "low",
        "iterations": None,
        "provider_opts": None,
    }


def test_agent_spec_identity_canonical_json_is_sorted_compact(planner: AgentSpec) -> None:
    s = planner.identity_canonical_json()
    assert " " not in s and "\n" not in s
    # First field after '{' should be 'iterations' (alphabetically first key)
    assert s.startswith('{"iterations":')


def test_two_equal_agent_specs_have_byte_identical_canonical_json() -> None:
    a = AgentSpec(role="planner", provider="openai", model="gpt-4.1-mini")
    b = AgentSpec(role="planner", provider="openai", model="gpt-4.1-mini")
    assert a.identity_canonical_json() == b.identity_canonical_json()


def test_chain_inputs_descriptor_sorts_input_keys() -> None:
    d1 = chain_inputs_descriptor(
        project="P",
        bug_id=1,
        dataset="defects4j",
        sr_model_id="x",
        candidate_source="y",
        input_keys=("b", "a"),
    )
    d2 = chain_inputs_descriptor(
        project="P",
        bug_id="1",
        dataset="defects4j",
        sr_model_id="x",
        candidate_source="y",
        input_keys=("a", "b"),
    )
    assert d1 == d2
    assert d1["input_keys"] == ["a", "b"]
    assert d1["bug_id"] == "1"  # int → str


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


def test_chain_hash_is_32_hex_chars_and_deterministic(
    tmp_path: Path, inputs: dict, planner: AgentSpec, tool_caller: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    h1 = store.chain_hash(inputs, [planner, tool_caller])
    h2 = store.chain_hash(inputs, [planner, tool_caller])
    assert h1 == h2
    assert len(h1) == 32
    assert all(c in "0123456789abcdef" for c in h1)


def test_chain_hash_differs_when_inputs_or_chain_change(
    tmp_path: Path, inputs: dict, planner: AgentSpec, tool_caller: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    base = store.chain_hash(inputs, [planner, tool_caller])
    other_chain = store.chain_hash(inputs, [planner])
    assert base != other_chain

    other_inputs = dict(inputs)
    other_inputs["sr_model_id"] = "different_sr"
    assert store.chain_hash(other_inputs, [planner, tool_caller]) != base


def test_chain_hash_accepts_dict_or_agent_spec(
    tmp_path: Path, inputs: dict, planner: AgentSpec
) -> None:
    store = CheckpointStore(tmp_path)
    a = store.chain_hash(inputs, [planner])
    b = store.chain_hash(inputs, [planner.identity()])
    assert a == b


def test_lookup_returns_none_when_missing(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert store.lookup("deadbeef" * 4) is None


def test_save_then_lookup_round_trips(tmp_path: Path, inputs: dict, planner: AgentSpec) -> None:
    store = CheckpointStore(tmp_path)
    input_list: list[dict[str, Any]] = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "go"},
        {"id": "rs_1", "type": "reasoning", "summary": []},
    ]
    messages = [{"role": "system", "content": "hi"}, {"role": "user", "content": "go"}]
    raw_responses = [{"raw": "stuff"}]
    path = store.save(
        inputs_descriptor=inputs,
        completed_chain=[planner],
        input_list=input_list,
        messages=messages,
        raw_responses=raw_responses,
    )
    assert path.exists()
    assert not (tmp_path / f"{path.stem}.tmp.json").exists(), "tmp file must be replaced atomically"

    digest = store.chain_hash(inputs, [planner])
    ckpt = store.lookup(digest)
    assert isinstance(ckpt, CompletedCheckpoint)
    assert ckpt.input_list == input_list
    assert ckpt.messages == messages
    assert ckpt.raw_responses == raw_responses
    assert ckpt.completed_chain == [planner.identity()]
    assert ckpt.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert ckpt.finished_at > 0


def test_find_longest_prefix_returns_zero_on_empty_store(
    tmp_path: Path,
    inputs: dict,
    planner: AgentSpec,
    tool_caller: AgentSpec,
    finisher: AgentSpec,
) -> None:
    store = CheckpointStore(tmp_path)
    k, ckpt = store.find_longest_prefix(inputs, [planner, tool_caller, finisher])
    assert k == 0
    assert ckpt is None


def test_find_longest_prefix_picks_longest_cached_match(
    tmp_path: Path,
    inputs: dict,
    planner: AgentSpec,
    tool_caller: AgentSpec,
    finisher: AgentSpec,
) -> None:
    store = CheckpointStore(tmp_path)
    # Save checkpoints for [planner] and [planner, tool_caller], but NOT
    # for the full chain [planner, tool_caller, finisher].
    store.save(
        inputs_descriptor=inputs,
        completed_chain=[planner],
        input_list=[{"role": "system", "content": "after planner"}],
        messages=[{"role": "system", "content": "after planner"}],
        raw_responses=[],
    )
    store.save(
        inputs_descriptor=inputs,
        completed_chain=[planner, tool_caller],
        input_list=[
            {"role": "system", "content": "after planner"},
            {"role": "user", "content": "after tool_caller"},
        ],
        messages=[
            {"role": "system", "content": "after planner"},
            {"role": "user", "content": "after tool_caller"},
        ],
        raw_responses=[],
    )

    k, ckpt = store.find_longest_prefix(inputs, [planner, tool_caller, finisher])
    assert k == 2
    assert ckpt is not None
    assert len(ckpt.input_list) == 2
    assert len(ckpt.messages) == 2


def test_find_longest_prefix_recognises_fully_complete_chain(
    tmp_path: Path,
    inputs: dict,
    planner: AgentSpec,
    tool_caller: AgentSpec,
    finisher: AgentSpec,
) -> None:
    store = CheckpointStore(tmp_path)
    full = [planner, tool_caller, finisher]
    store.save(
        inputs_descriptor=inputs,
        completed_chain=full,
        input_list=[{"role": "assistant", "content": "done"}],
        messages=[{"role": "assistant", "content": "done"}],
        raw_responses=[],
    )
    k, ckpt = store.find_longest_prefix(inputs, full)
    assert k == 3
    assert ckpt is not None


def test_lookup_skips_old_schema_version_as_cache_miss(tmp_path: Path) -> None:
    """Schema-v1 checkpoints (pre input_list refactor) are treated as misses."""
    store = CheckpointStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    digest = "f" * 32
    (tmp_path / f"{digest}.json").write_text(
        '{"schema_version": 1, "completed_chain": [], "messages": [], '
        '"raw_responses": [], "finished_at": 0.0}',
        encoding="utf-8",
    )
    assert store.lookup(digest) is None


def test_lookup_skips_future_schema_version_as_cache_miss(tmp_path: Path) -> None:
    """Schema-vN-unknown checkpoints are also misses (degrade gracefully)."""
    store = CheckpointStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    digest = "e" * 32
    (tmp_path / f"{digest}.json").write_text(
        '{"schema_version": 99, "completed_chain": [], "input_list": [], '
        '"messages": [], "raw_responses": [], "finished_at": 0.0}',
        encoding="utf-8",
    )
    assert store.lookup(digest) is None
