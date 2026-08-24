"""Local registry of OpenAI batches sent for Agent4LR, plus the custom_id codec.

One JSON file per batch under ``data/batches/agent4lr/`` (see
:func:`src.common.config.get_lr_batch_registry_dir`), written atomically.
Uploaded request JSONLs are archived alongside as
``<batch_id>.input.jsonl`` and downloaded outputs as
``<batch_id>.output.jsonl`` / ``<batch_id>.errors.jsonl`` — OpenAI
expires result files after ~30 days, so the archive is the durable copy.

The registry is deliberately minimal: no per-config subscriber tracking.
The tick re-derives all work from CLI flags + checkpoint state; the
registry only answers "which batches haven't been consumed yet" and
"which requests are currently in flight".

custom_id format (Batch API limit: 64 chars)::

    <BM>:<project>:<bug_id>:<slot_hash>:<call_index>
    e.g. D4J:JacksonDatabind:112:9f2c...e1:07

``<BM>`` is the canonical short benchmark code from
:func:`src.core.layout.normalize_benchmark_name` (``D4J``/``BIP``);
``slot_hash`` is the 32-hex pending-state key; ``call_index`` is the
state's ``calls_made`` at send time (2 digits), making stale responses
detectable after a state has advanced through another path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from src.core.layout import normalize_benchmark_name

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 1
CUSTOM_ID_MAX_LEN = 64

# Reverse of BENCHMARK_ALIASES restricted to the adapter keys used as
# ``dataset`` throughout the codebase (aliases like "d4j" also map to
# "D4J", so pick the canonical long form explicitly).
_CANONICAL_DATASET_BY_CODE: dict[str, str] = {
    "D4J": "defects4j",
    "BIP": "bugsinpy",
}


def dataset_for_benchmark_code(code: str) -> str:
    """Map a custom_id benchmark code (``D4J``) back to the adapter key."""
    try:
        return _CANONICAL_DATASET_BY_CODE[code]
    except KeyError:
        raise ValueError(
            f"Unknown benchmark code {code!r} in custom_id; "
            f"known: {sorted(_CANONICAL_DATASET_BY_CODE)}"
        ) from None


@dataclass(frozen=True)
class RequestKey:
    """Decoded form of one batch request's ``custom_id``."""

    benchmark: str  # canonical short code, e.g. "D4J"
    project: str
    bug_id: str
    slot_hash: str
    call_index: int

    @classmethod
    def for_state(
        cls, *, dataset: str, project: str, bug_id: str, slot_hash: str, call_index: int
    ) -> RequestKey:
        """Build a key from tick-side values (``dataset`` is the adapter key)."""
        return cls(
            benchmark=normalize_benchmark_name(dataset),
            project=project,
            bug_id=str(bug_id),
            slot_hash=slot_hash,
            call_index=call_index,
        )

    @property
    def dataset(self) -> str:
        """The adapter key (``defects4j``/``bugsinpy``) for path resolution."""
        return dataset_for_benchmark_code(self.benchmark)

    def encode(self) -> str:
        """Serialise to the ``custom_id`` string (asserts the 64-char budget)."""
        for part, label in ((self.project, "project"), (self.bug_id, "bug_id")):
            if ":" in part:
                raise ValueError(f"{label} {part!r} must not contain ':'")
        encoded = (
            f"{self.benchmark}:{self.project}:{self.bug_id}:{self.slot_hash}:{self.call_index:02d}"
        )
        if len(encoded) > CUSTOM_ID_MAX_LEN:
            raise ValueError(
                f"custom_id {encoded!r} exceeds {CUSTOM_ID_MAX_LEN} chars ({len(encoded)})"
            )
        return encoded

    @classmethod
    def decode(cls, custom_id: str) -> RequestKey:
        parts = custom_id.split(":")
        if len(parts) != 5:
            raise ValueError(f"Malformed custom_id {custom_id!r}: expected 5 ':'-separated parts")
        benchmark, project, bug_id, slot_hash, call_index = parts
        return cls(
            benchmark=benchmark,
            project=project,
            bug_id=bug_id,
            slot_hash=slot_hash,
            call_index=int(call_index),
        )


@dataclass
class BatchRecord:
    """One sent batch as tracked locally."""

    batch_id: str
    input_file_id: str
    endpoint: str
    completion_window: str
    model: str
    n_requests: int
    custom_ids: list[str]
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    last_status: str = "validating"
    processed: bool = False
    output_file_id: str | None = None
    error_file_id: str | None = None
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> BatchRecord:
        data = json.loads(text)
        version = data.get("schema_version", 0)
        if version != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported batch record schema_version={version!r}")
        return cls(**data)


class BatchRegistry:
    """Directory of ``<batch_id>.json`` records with atomic writes."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, batch_id: str) -> Path:
        return self.root / f"{batch_id}.json"

    def archive_path(self, batch_id: str, kind: str) -> Path:
        """Path of an archived JSONL (``kind`` ∈ input/output/errors)."""
        return self.root / f"{batch_id}.{kind}.jsonl"

    def _write(self, record: BatchRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self._path(record.batch_id)
        tmp_path = final_path.with_suffix(".tmp.json")
        tmp_path.write_text(record.to_json(), encoding="utf-8")
        os.replace(tmp_path, final_path)
        return final_path

    def add(self, record: BatchRecord) -> Path:
        """Persist a freshly created batch (stamps ``created_at`` if unset)."""
        if not record.created_at:
            record = replace(record, created_at=time.time())
        path = self._write(record)
        logger.info(
            "Registered batch %s (%d requests, model=%s)",
            record.batch_id,
            record.n_requests,
            record.model,
        )
        return path

    def get(self, batch_id: str) -> BatchRecord | None:
        path = self._path(batch_id)
        if not path.exists():
            return None
        return BatchRecord.from_json(path.read_text(encoding="utf-8"))

    def all_records(self) -> list[BatchRecord]:
        """All records, oldest first."""
        if not self.root.exists():
            return []
        records = [
            BatchRecord.from_json(p.read_text(encoding="utf-8"))
            for p in sorted(self.root.glob("*.json"))
            if not p.name.endswith(".tmp.json")
        ]
        return sorted(records, key=lambda r: r.created_at)

    def unprocessed(self) -> list[BatchRecord]:
        """Records whose outputs haven't been consumed yet, oldest first."""
        return [r for r in self.all_records() if not r.processed]

    def mark(
        self,
        batch_id: str,
        *,
        last_status: str | None = None,
        processed: bool | None = None,
        output_file_id: str | None = None,
        error_file_id: str | None = None,
    ) -> BatchRecord:
        """Update a record's mutable fields and persist."""
        record = self.get(batch_id)
        if record is None:
            raise FileNotFoundError(f"No batch record {batch_id!r} under {self.root}")
        if last_status is not None:
            record.last_status = last_status
        if processed is not None:
            record.processed = processed
        if output_file_id is not None:
            record.output_file_id = output_file_id
        if error_file_id is not None:
            record.error_file_id = error_file_id
        self._write(record)
        return record

    def in_flight_custom_ids(self) -> set[str]:
        """Union of custom_ids across all unprocessed batches.

        This is the "already included in a pending batch" filter: any
        batch whose outputs haven't been consumed — whatever its remote
        status — blocks a resend of its requests. Failed/expired/
        cancelled batches become resendable as soon as the processor
        consumes them (marking ``processed=True``).
        """
        ids: set[str] = set()
        for record in self.unprocessed():
            ids.update(record.custom_ids)
        return ids


__all__ = [
    "CUSTOM_ID_MAX_LEN",
    "REGISTRY_SCHEMA_VERSION",
    "BatchRecord",
    "BatchRegistry",
    "RequestKey",
    "dataset_for_benchmark_code",
]
