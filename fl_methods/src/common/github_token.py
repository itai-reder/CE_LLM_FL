"""GitHub API token resolution (file-based).

The token lives in ``<PROJECT_ROOT>/.gh_token`` (gitignored), not in a
``GH_TOKEN``/``GITHUB_TOKEN`` environment variable, so shells and batch jobs
need no secret exported. When the file is missing or empty, callers fall back
to unauthenticated GitHub API requests (60 requests/hour).
"""

from __future__ import annotations

import logging

from src.common.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

GH_TOKEN_FILE = PROJECT_ROOT / ".gh_token"

_warned_missing = False


def get_github_token() -> str | None:
    """Return the token from ``.gh_token``, or None (warning once) when unavailable."""
    global _warned_missing
    try:
        token = GH_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if token:
        return token
    if not _warned_missing:
        _warned_missing = True
        logger.warning(
            "No GitHub token file at %s; GitHub API requests will be unauthenticated "
            "(60 requests/hour rate limit)",
            GH_TOKEN_FILE,
        )
    return None
