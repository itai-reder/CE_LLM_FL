"""Tests for the Agent4LR batch registry and custom_id codec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agent4lr.batch_registry import (
    CUSTOM_ID_MAX_LEN,
    BatchRecord,
    BatchRegistry,
    RequestKey,
    dataset_for_benchmark_code,
)

HASH32 = "9f2c" + "0" * 26 + "e1"


def _record(batch_id: str, custom_ids: list[str], **kwargs: Any) -> BatchRecord:
    defaults: dict[str, Any] = dict(
        batch_id=batch_id,
        input_file_id=f"file-{batch_id}",
        endpoint="/v1/responses",
        completion_window="24h",
        model="gpt-5-mini",
        n_requests=len(custom_ids),
        custom_ids=custom_ids,
    )
    defaults.update(kwargs)
    return BatchRecord(**defaults)


class TestRequestKey:
    def test_encode_decode_round_trip(self) -> None:
        key = RequestKey.for_state(
            dataset="defects4j", project="Lang", bug_id="1", slot_hash=HASH32, call_index=7
        )
        encoded = key.encode()
        assert encoded == f"D4J:Lang:1:{HASH32}:07"
        assert RequestKey.decode(encoded) == key
        assert key.dataset == "defects4j"

    def test_worst_case_fits_64_chars(self) -> None:
        # Longest real project name (BugsInPy's JacksonDatabind-style) + 3-digit bug.
        key = RequestKey.for_state(
            dataset="bugsinpy",
            project="JacksonDatabind",
            bug_id="112",
            slot_hash=HASH32,
            call_index=99,
        )
        assert len(key.encode()) <= CUSTOM_ID_MAX_LEN

    def test_encode_rejects_colon_in_parts(self) -> None:
        key = RequestKey(
            benchmark="D4J", project="La:ng", bug_id="1", slot_hash=HASH32, call_index=0
        )
        with pytest.raises(ValueError, match="must not contain"):
            key.encode()

    def test_decode_rejects_malformed(self) -> None:
        with pytest.raises(ValueError, match="Malformed custom_id"):
            RequestKey.decode("D4J:Lang:1:tooshort")

    def test_unknown_benchmark_code_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark code"):
            dataset_for_benchmark_code("XYZ")


class TestBatchRegistry:
    def test_add_get_round_trip(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        registry.add(_record("batch-1", ["a", "b"]))
        record = registry.get("batch-1")
        assert record is not None
        assert record.custom_ids == ["a", "b"]
        assert record.created_at > 0
        assert not record.processed

    def test_mark_updates_fields(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        registry.add(_record("batch-1", ["a"]))
        registry.mark("batch-1", last_status="completed", processed=True, output_file_id="f-out")
        record = registry.get("batch-1")
        assert record is not None
        assert record.last_status == "completed"
        assert record.processed is True
        assert record.output_file_id == "f-out"

    def test_mark_missing_raises(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        with pytest.raises(FileNotFoundError):
            registry.mark("nope", last_status="completed")

    def test_in_flight_ignores_processed(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        registry.add(_record("batch-1", ["a", "b"]))
        registry.add(_record("batch-2", ["c"], last_status="failed"))
        registry.add(_record("batch-3", ["d"]))
        # Whatever the remote status, unprocessed batches block resends...
        assert registry.in_flight_custom_ids() == {"a", "b", "c", "d"}
        # ...and consuming a batch releases its requests.
        registry.mark("batch-2", processed=True)
        registry.mark("batch-3", processed=True)
        assert registry.in_flight_custom_ids() == {"a", "b"}

    def test_unprocessed_oldest_first(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        registry.add(_record("batch-b", ["x"], created_at=200.0))
        registry.add(_record("batch-a", ["y"], created_at=100.0))
        assert [r.batch_id for r in registry.unprocessed()] == ["batch-a", "batch-b"]

    def test_empty_registry(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        assert registry.all_records() == []
        assert registry.in_flight_custom_ids() == set()
        assert registry.get("nope") is None

    def test_schema_version_mismatch_raises(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path / "reg")
        registry.add(_record("batch-1", ["a"]))
        path = tmp_path / "reg" / "batch-1.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 0'),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version"):
            registry.get("batch-1")
