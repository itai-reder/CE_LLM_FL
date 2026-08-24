"""End-to-end tests for the Agent4LR batch tick over a fake OpenAI backend.

Drives ``run_tick`` through a full multi-round lifecycle on a ``tmp_path``
processed tree: planner round → mid-slot tool_caller rounds (pending
state persisted between ticks) → finisher divergence of two
prefix-sharing configs → zero-LLM finalization. Plus the failure-path
scenarios: in-flight blocking, failed/expired lines leaving pending
states for deterministic resend, stale call_index and
checkpoint-race drops, ollama rejection, and dry-run purity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent4lr.agents import AgentSpec
from src.agent4lr.batch_registry import BatchRegistry, RequestKey
from src.agent4lr.batch_runner import run_tick, validate_openai_only
from src.agent4lr.checkpoints import CheckpointStore, PendingStateStore

# Exactly 20 lines (duplicates allowed) — FlexFL's combination rule, enforced
# by the candidate_list_skip_reason gate in run_tick.
CANDIDATES = ["a.b.X.foo(int)", "y.z.Other.bar()"] * 10
CORPUS_METHODS = ["a.b$X.foo(int)", "y.z$Other.bar()"]
CORPUS_CODES = ["int foo() { return 1; }", "void bar() {}"]

PLANNER = AgentSpec(role="planner", provider="openai", model="gpt-5-mini")
TOOL = AgentSpec(role="tool_caller", provider="openai", model="gpt-5-nano", iterations=2)
FIN_A = AgentSpec(role="finisher", provider="openai", model="gpt-5-mini")
FIN_B = AgentSpec(role="finisher", provider="openai", model="gpt-5-mini", reasoning_effort="medium")

CONFIGS = {
    "cfg-A": (PLANNER, TOOL, FIN_A),
    "cfg-B": (PLANNER, TOOL, FIN_B),
}


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeBatchBackend:
    """In-memory Files/Batches API with scripted completion."""

    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self._n = 0

    def upload_jsonl(self, content: bytes, filename: str) -> str:
        self._n += 1
        file_id = f"file-in-{self._n}"
        self.uploads[file_id] = content
        return file_id

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        self._n += 1
        batch_id = f"batch-{self._n}"
        self.batches[batch_id] = {
            "id": batch_id,
            "status": "in_progress",
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "metadata": metadata,
        }
        return dict(self.batches[batch_id])

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        return dict(self.batches[batch_id])

    def download_file(self, file_id: str) -> bytes:
        return self.files[file_id]

    # -- test helpers -------------------------------------------------------

    def request_lines(self, batch_id: str) -> list[dict[str, Any]]:
        content = self.uploads[self.batches[batch_id]["input_file_id"]]
        return [json.loads(line) for line in content.decode("utf-8").splitlines() if line]

    def complete(
        self,
        batch_id: str,
        output_lines: list[dict[str, Any]],
        *,
        status: str = "completed",
        error_lines: list[dict[str, Any]] | None = None,
    ) -> None:
        record = self.batches[batch_id]
        record["status"] = status
        if output_lines:
            file_id = f"file-out-{batch_id}"
            self.files[file_id] = (
                "\n".join(json.dumps(line) for line in output_lines) + "\n"
            ).encode("utf-8")
            record["output_file_id"] = file_id
        if error_lines:
            file_id = f"file-err-{batch_id}"
            self.files[file_id] = (
                "\n".join(json.dumps(line) for line in error_lines) + "\n"
            ).encode("utf-8")
            record["error_file_id"] = file_id


def _output_line(
    custom_id: str,
    *,
    content: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    status_code: int = 200,
    body_status: str = "completed",
    error: Any = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if content:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    for i, (name, args) in enumerate(tool_calls or []):
        output.append(
            {
                "type": "function_call",
                "call_id": f"{custom_id}-c{i}",
                "name": name,
                "arguments": json.dumps(args),
            }
        )
    return {
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": {"status": body_status, "output": output}},
        "error": error,
    }


# ---------------------------------------------------------------------------
# Environment fixture: fake processed tree + patched path resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    processed_root = tmp_path / "processed"
    bug_dir = processed_root / "Lang" / "1"
    bug_dir.mkdir(parents=True)
    (bug_dir / "bug_report.json").write_text(
        json.dumps({"title": "NPE in foo", "description": "foo NPEs on negatives"}),
        encoding="utf-8",
    )
    (bug_dir / "trigger_test_clean.txt").write_text("public void testFoo() {}", encoding="utf-8")
    (bug_dir / "faults.csv").write_text(
        "path,line,signature\na/b/X.java,1,a.b$X.foo(int)\n", encoding="utf-8"
    )
    top20 = bug_dir / "FlexFL" / "SR" / "rankings" / "top20" / "llama3.1_8b.txt"
    top20.parent.mkdir(parents=True)
    top20.write_text("\n".join(CANDIDATES) + "\n", encoding="utf-8")

    def fake_processed_dir(project: str, bug_id: str | int, dataset: str = "defects4j") -> Path:
        return processed_root / project / str(bug_id)

    def fake_candidate_file(
        project: str, bug_id: str | int, *, sr_model_id: str, dataset: str = "defects4j"
    ) -> Path:
        return (
            fake_processed_dir(project, bug_id)
            / "FlexFL"
            / "SR"
            / "rankings"
            / "top20"
            / f"{sr_model_id}.txt"
        )

    monkeypatch.setattr("src.agent4lr.batch_runner.get_processed_dir", fake_processed_dir)
    monkeypatch.setattr("src.agent4lr.io.get_processed_dir", fake_processed_dir)
    monkeypatch.setattr("src.agent4lr.io.get_lr_candidate_file", fake_candidate_file)
    monkeypatch.setattr("src.agent4lr.agent.get_processed_dir", fake_processed_dir)
    monkeypatch.setattr("src.agent4lr.batch_runner.load_tracker", lambda *a, **k: {"lr": {}})
    from src.agent4sr import function_call as sr_fc

    monkeypatch.setattr(sr_fc, "load_corpus_methods", lambda *a, **k: list(CORPUS_METHODS))
    monkeypatch.setattr(sr_fc, "load_corpus_codes", lambda *a, **k: list(CORPUS_CODES))

    ckpt_dir = bug_dir / "FlexFL" / "LR" / "checkpoints"
    return {
        "backend": FakeBatchBackend(),
        "registry": BatchRegistry(tmp_path / "batches"),
        "bug_dir": bug_dir,
        "ckpt_store": CheckpointStore(ckpt_dir),
        "pending_store": PendingStateStore.for_checkpoints_dir(ckpt_dir),
        "finalized_calls": [],
    }


def _tick(env: dict[str, Any], **overrides: Any) -> Any:
    def record_finalize(**kwargs: Any) -> dict[str, Any]:
        env["finalized_calls"].append(kwargs)
        return {"status": "OK"}

    kwargs: dict[str, Any] = dict(
        dataset="defects4j",
        bugs=[("Lang", "1")],
        configs=dict(CONFIGS),
        sr_model_id="llama3.1_8b",
        input_keys=("bug_report", "trigger_test"),
        registry=env["registry"],
        backend=env["backend"],
        finalize_fn=record_finalize,
    )
    kwargs.update(overrides)
    return run_tick(**kwargs)


# ---------------------------------------------------------------------------
# The full lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_two_prefix_sharing_configs(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    registry: BatchRegistry = env["registry"]
    pending: PendingStateStore = env["pending_store"]
    ckpts: CheckpointStore = env["ckpt_store"]

    # --- Tick 1: one deduped planner request for both configs ---------------
    report = _tick(env)
    assert len(report.requests) == 1
    assert report.requests[0].config_names == ("cfg-A", "cfg-B")
    assert len(report.batches_created) == 1
    planner_batch = report.batches_created[0]
    lines = backend.request_lines(planner_batch)
    assert len(lines) == 1
    planner_cid = lines[0]["custom_id"]
    assert planner_cid.startswith("D4J:Lang:1:") and planner_cid.endswith(":00")
    assert lines[0]["url"] == "/v1/responses"
    assert lines[0]["body"]["model"] == "gpt-5-mini"
    assert lines[0]["body"]["tool_choice"] == "none"
    # Pending state was persisted before send (the custom_id → state binding).
    key = RequestKey.decode(planner_cid)
    assert pending.load(key.slot_hash) is not None

    # --- Tick 2 (batch still running): in-flight filter blocks a resend -----
    report = _tick(env)
    assert report.requests == []
    assert report.batches_created == []
    assert report.in_flight == [planner_cid]

    # --- Tick 3: planner completes → checkpoint; tool round 1 sent ----------
    backend.complete(planner_batch, [_output_line(planner_cid, content="the plan")])
    report = _tick(env)
    assert report.processed_batches == [planner_batch]
    assert report.outcomes == {"slot_completed": 1}
    assert ckpts.lookup(key.slot_hash) is not None  # planner checkpoint
    assert pending.load(key.slot_hash) is None  # promoted, pending gone
    assert len(report.requests) == 1  # tool_caller round 1, still shared
    tool_batch = report.batches_created[0]
    (tool_line,) = backend.request_lines(tool_batch)
    tool_cid = tool_line["custom_id"]
    assert tool_cid.endswith(":00")
    assert tool_line["body"]["input"][-1]["content"] == "You have 2 tool calls remaining."
    assert tool_line["body"]["tool_choice"] == "required"

    # --- Tick 4: snippet call → mid-slot pending survives; round 2 sent -----
    backend.complete(
        tool_batch,
        [_output_line(tool_cid, tool_calls=[("get_snippet_of_method", {"method_number": 1})])],
    )
    report = _tick(env)
    assert report.outcomes == {"applied": 1}
    tool_key = RequestKey.decode(tool_cid)
    mid_state = pending.load(tool_key.slot_hash)
    assert mid_state is not None and mid_state.calls_made == 1
    assert any(
        item.get("type") == "function_call_output" and "int foo()" in item.get("output", "")
        for item in mid_state.input_list
    )
    (round2_line,) = backend.request_lines(report.batches_created[0])
    round2_cid = round2_line["custom_id"]
    assert round2_cid.endswith(":01")  # call_index advanced
    assert round2_line["body"]["input"][-1]["content"] == "You have 1 tool calls remaining."

    # --- Tick 5: exit → tool checkpoint; finishers diverge into 2 requests --
    backend.complete(
        report.batches_created[0], [_output_line(round2_cid, tool_calls=[("exit", {})])]
    )
    report = _tick(env)
    assert report.outcomes == {"slot_completed": 1}
    assert len(report.requests) == 2  # cfg-A and cfg-B finisher specs differ
    assert {r.config_names for r in report.requests} == {("cfg-A",), ("cfg-B",)}
    # Same model → one grouped batch carrying both finisher requests.
    assert len(report.batches_created) == 1
    fin_lines = backend.request_lines(report.batches_created[0])
    assert len(fin_lines) == 2
    bodies_by_cid = {line["custom_id"]: line["body"] for line in fin_lines}
    assert any("reasoning" in body for body in bodies_by_cid.values())  # cfg-B medium effort
    assert any("reasoning" not in body for body in bodies_by_cid.values())  # cfg-A default

    # --- Tick 6: finisher responses → both chains finalize ------------------
    backend.complete(
        report.batches_created[0],
        [
            _output_line(cid, tool_calls=[("rank_methods", {"top_5_methods": [1, 2, 1, 2, 1]})])
            for cid in bodies_by_cid
        ],
    )
    report = _tick(env)
    assert report.outcomes == {"slot_completed": 2}
    assert sorted(c for _, _, c in report.finalized) == ["cfg-A", "cfg-B"]
    assert len(env["finalized_calls"]) == 2
    assert {c["config_name"] for c in env["finalized_calls"]} == {"cfg-A", "cfg-B"}
    assert report.requests == []
    assert report.batches_created == []
    # Registry fully drained.
    assert registry.in_flight_custom_ids() == set()


# ---------------------------------------------------------------------------
# Failure-path scenarios
# ---------------------------------------------------------------------------


def test_failed_line_leaves_pending_for_deterministic_resend(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    report = _tick(env)
    batch_id = report.batches_created[0]
    (line,) = backend.request_lines(batch_id)
    cid = line["custom_id"]

    backend.complete(batch_id, [_output_line(cid, content="x", status_code=500)])
    report = _tick(env)
    assert report.outcomes == {"request_failed": 1}
    # The state never advanced, so the tick resends the same custom_id...
    assert [item.key.encode() for item in report.requests] == [cid]
    # ...with a byte-identical request body.
    (resent,) = backend.request_lines(report.batches_created[0])
    assert resent["body"] == line["body"]


def test_expired_batch_partial_output_consumed_rest_resent(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    # Two bugs → two planner requests in one batch.
    bug2 = env["bug_dir"].parent / "2"
    import shutil

    shutil.copytree(env["bug_dir"], bug2)
    report = _tick(env, bugs=[("Lang", "1"), ("Lang", "2")])
    batch_id = report.batches_created[0]
    lines = backend.request_lines(batch_id)
    assert len(lines) == 2
    done_cid, lost_cid = lines[0]["custom_id"], lines[1]["custom_id"]

    # Expired with partial output: only the first request completed.
    backend.complete(batch_id, [_output_line(done_cid, content="plan")], status="expired")
    report = _tick(env, bugs=[("Lang", "1"), ("Lang", "2")])
    assert report.outcomes == {"slot_completed": 1}
    # The finished bug moved on to its tool round; the lost one resent as-is.
    cids = sorted(item.key.encode() for item in report.requests)
    assert lost_cid in cids
    assert len(cids) == 2
    assert all(not r.processed for r in env["registry"].unprocessed())


def test_stale_call_index_dropped(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    report = _tick(env)
    batch_id = report.batches_created[0]
    (line,) = backend.request_lines(batch_id)
    cid = line["custom_id"]
    stale_cid = cid[:-2] + "05"  # claims call_index 5; state is at 0
    backend.complete(batch_id, [_output_line(stale_cid, content="late")])
    report = _tick(env)
    assert report.outcomes == {"stale": 1}


def test_response_after_checkpoint_race_dropped(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    ckpts: CheckpointStore = env["ckpt_store"]
    pending: PendingStateStore = env["pending_store"]

    report = _tick(env)
    batch_id = report.batches_created[0]
    (line,) = backend.request_lines(batch_id)
    cid = line["custom_id"]
    key = RequestKey.decode(cid)

    # A sync run completes the slot first (content-addressed same hash).
    state = pending.load(key.slot_hash)
    assert state is not None
    ckpts.save(
        inputs_descriptor=state.inputs_descriptor,
        completed_chain=[*state.completed_chain, state.slot_spec],
        input_list=state.input_list,
        messages=state.messages,
        raw_responses=state.raw_responses,
    )
    backend.complete(batch_id, [_output_line(cid, content="duplicate work")])
    report = _tick(env)
    assert report.outcomes == {"already_complete": 1}
    assert pending.load(key.slot_hash) is None  # cleaned up


def test_missing_pending_state_dropped(env: dict[str, Any]) -> None:
    backend: FakeBatchBackend = env["backend"]
    pending: PendingStateStore = env["pending_store"]
    report = _tick(env)
    batch_id = report.batches_created[0]
    (line,) = backend.request_lines(batch_id)
    cid = line["custom_id"]
    pending.delete(RequestKey.decode(cid).slot_hash)  # user abandoned the request
    backend.complete(batch_id, [_output_line(cid, content="orphan")])
    report = _tick(env)
    assert report.outcomes == {"missing_pending": 1}


def test_ollama_config_rejected() -> None:
    bad: dict[str, tuple[AgentSpec, ...]] = {
        "mixed": (
            PLANNER,
            AgentSpec(role="tool_caller", provider="ollama", model="llama3.1:8b", iterations=10),
            FIN_A,
        )
    }
    with pytest.raises(ValueError, match="openai-provider slots only"):
        validate_openai_only(bad)


def test_dry_run_writes_nothing(env: dict[str, Any], tmp_path: Path) -> None:
    report = _tick(env, backend=None, finalize_fn=None, dry_run=True)
    assert len(report.requests) == 1
    assert report.batches_created == []
    assert not (tmp_path / "batches").exists()
    assert not env["pending_store"].root.exists()
    assert not env["ckpt_store"].root.exists()


def test_max_requests_chunks_batches(env: dict[str, Any]) -> None:
    import shutil

    bug2 = env["bug_dir"].parent / "2"
    shutil.copytree(env["bug_dir"], bug2)
    report = _tick(env, bugs=[("Lang", "1"), ("Lang", "2")], max_requests=1)
    assert len(report.requests) == 2
    assert len(report.batches_created) == 2  # one request per batch


def test_existing_results_skip_via_tracker(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tracker knows this identity as "cfg-A" and both result files exist.
    model_dir = env["bug_dir"] / "FlexFL" / "LR" / "Agent4LR" / "cfg-A"
    model_dir.mkdir(parents=True)
    (model_dir / "lr_result.json").write_text("{}", encoding="utf-8")
    (model_dir / "top5.txt").write_text("", encoding="utf-8")

    def fake_tracker(*a: Any, **k: Any) -> dict[str, Any]:
        return {
            "lr": {
                "cfg-A": {
                    "config_name": "cfg-A",
                    "agent_chain": [s.identity() for s in CONFIGS["cfg-A"]],
                    "sr_model_id": "llama3.1_8b",
                    "candidate_source": "FlexFL/SR/rankings/top20/llama3.1_8b.txt",
                    "input": ["bug_report", "trigger_test"],
                }
            }
        }

    monkeypatch.setattr("src.agent4lr.batch_runner.load_tracker", fake_tracker)
    report = _tick(env)
    skip_reasons = {c: r for _, _, c, r in report.skipped}
    assert "cfg-A" in skip_reasons and "results exist" in skip_reasons["cfg-A"]
    # cfg-B still proceeds (planner request built for it alone).
    assert len(report.requests) == 1
    assert report.requests[0].config_names == ("cfg-B",)
