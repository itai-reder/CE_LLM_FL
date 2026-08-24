"""OpenAI Files/Batches client for the Agent4LR batch workflow.

Thin wrapper over the four API surfaces the tick needs, behind the
:class:`BatchBackend` protocol so tests can inject a fake. Mirrors
:class:`src.agent4lr.providers.openai.OpenAIProvider`'s conventions:
lazy ``openai`` import and ``OPENAI_API_KEY`` fallback for the key.

Batch API facts the runner relies on (see the OpenAI Batch guide):

* one model per input file — the tick groups requests by model;
* ``/v1/responses`` is a supported batch endpoint;
* batches complete within the ``completion_window`` (24h) or expire,
  and expired/cancelled batches still expose partial results via
  ``output_file_id``;
* output line order is unrelated to input order — correlate by
  ``custom_id``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

BATCH_ENDPOINT = "/v1/responses"
DEFAULT_COMPLETION_WINDOW = "24h"

# Remote statuses in which a batch may still make progress; everything
# else is terminal and its (possibly partial) outputs can be consumed.
ACTIVE_BATCH_STATUSES = frozenset({"validating", "in_progress", "finalizing", "cancelling"})


class BatchBackend(Protocol):
    """The four OpenAI API surfaces the batch tick needs."""

    def upload_jsonl(self, content: bytes, filename: str) -> str:
        """Upload a request JSONL with ``purpose="batch"``; return the file id."""
        ...

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        """Create a batch; return the Batch object as a plain dict."""
        ...

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        """Fetch the current Batch object as a plain dict."""
        ...

    def download_file(self, file_id: str) -> bytes:
        """Download a result/error file's raw bytes."""
        ...


def _to_dict(obj: Any) -> dict[str, Any]:
    return obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)


class OpenAIBatchBackend:
    """Live implementation of :class:`BatchBackend` over the openai SDK."""

    def __init__(self, *, api_key: str | None = None, timeout: float = 300.0) -> None:
        try:
            import openai  # noqa: F401 — lazy import; surfaces ImportError if missing
        except ImportError as exc:  # pragma: no cover — covered by user environment
            raise ImportError(
                "openai SDK is required for OpenAIBatchBackend. "
                "Install via `pip install openai` in the cefl conda env."
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAIBatchBackend needs an API key. Pass --openai-api-key on "
                "the CLI or set the OPENAI_API_KEY env var."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=timeout)

    def upload_jsonl(self, content: bytes, filename: str) -> str:
        file_obj = self._client.files.create(file=(filename, content), purpose="batch")
        file_id = str(_to_dict(file_obj)["id"])
        logger.info("Uploaded batch input %s (%d bytes) as %s", filename, len(content), file_id)
        return file_id

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        # The SDK types endpoint/completion_window as Literals; the protocol
        # keeps plain str so fakes stay simple. Values are validated server-side.
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=cast(Any, endpoint),
            completion_window=cast(Any, completion_window),
            metadata=metadata,
        )
        return _to_dict(batch)

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        return _to_dict(self._client.batches.retrieve(batch_id))

    def download_file(self, file_id: str) -> bytes:
        response = self._client.files.content(file_id)
        content = response.read()
        return content if isinstance(content, bytes) else bytes(content)


__all__ = [
    "ACTIVE_BATCH_STATUSES",
    "BATCH_ENDPOINT",
    "DEFAULT_COMPLETION_WINDOW",
    "BatchBackend",
    "OpenAIBatchBackend",
]
