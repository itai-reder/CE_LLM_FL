"""Bug report fetching for Defects4J projects.

Queries the Defects4J container for bug report URLs and parses them using
:mod:`src.extraction.report_parser`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.common.config import BIP_INDEX_CSV
from src.common.github_token import get_github_token
from src.core.layout import normalize_benchmark_name
from src.extraction.d4j import D4JRepo
from src.extraction.docker import run_defects4j
from src.extraction.report_parser import parse_report

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting per domain
# ---------------------------------------------------------------------------

DOMAIN_BUFFER_TIME = 3  # seconds between calls to the same domain
_last_url_call: dict[str, float] = {}


def _parse_domain(url: str) -> str | None:
    match = re.match(r"https?://([^/]+)", url)
    return match.group(1) if match else None


def _wait_for_domain(url: str) -> None:
    domain = _parse_domain(url)
    if domain is None:
        return
    last_call = _last_url_call.get(domain, 0.0)
    elapsed = time.time() - last_call
    if elapsed < DOMAIN_BUFFER_TIME:
        wait = DOMAIN_BUFFER_TIME - elapsed
        logger.debug("Waiting %.2f seconds before calling %s", wait, domain)
        time.sleep(wait)
    _last_url_call[domain] = time.time()


# ---------------------------------------------------------------------------
# URL lookup
# ---------------------------------------------------------------------------


def get_bug_report_url(project: str, bug_id: int) -> str | None:
    """Query the container for the bug report URL.

    Returns None if the URL is ``UNKNOWN`` or missing.
    """
    result = run_defects4j(
        ["query", "-p", project, "-q", "report.url"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            "defects4j query failed for %s-%d: %s",
            project,
            bug_id,
            result.stderr.strip(),
        )
        return None

    for line in result.stdout.strip().splitlines():
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0] == str(bug_id):
            url = parts[1].strip()
            if url and url != "UNKNOWN":
                return url
    return None


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------


def get_bug_report(project: str, bug_id: int) -> dict[str, str] | None:
    """Fetch and parse the bug report for *project*/*bug_id*.

    Returns a dict with ``url``, ``title``, ``description``, ``raw``
    (or ``error`` on failure).  Returns None if no URL is available.
    """
    url = get_bug_report_url(project, bug_id)
    if url is None:
        return None

    _wait_for_domain(url)
    report = parse_report(url)
    return report


# ---------------------------------------------------------------------------
# BugsInPy: GitHub commit -> issue -> report (offline-first via cached JSON)
# ---------------------------------------------------------------------------


def _bip_github_url(project: str) -> str | None:
    """Read ``github_url`` from a BugsInPy project's ``project.info``."""
    info = BIP_INDEX_CSV.parent / project / "project.info"
    if not info.exists():
        return None
    for line in info.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("github_url"):
            return stripped.split("=", 1)[1].strip().strip('"').strip()
    return None


def _owner_repo(github_url: str) -> tuple[str, str] | None:
    match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", github_url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _github_commit_message(owner: str, repo: str, commit: str) -> str | None:
    """Fetch a commit message via the GitHub API (honors the ``.gh_token`` file)."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = get_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    _wait_for_domain(api_url)
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        message = response.json().get("commit", {}).get("message")
        return str(message) if message else None
    except requests.RequestException as exc:
        logger.warning("GitHub commit fetch failed for %s/%s@%s: %s", owner, repo, commit, exc)
        return None


def _minimal_bip_report(github_url: str, commit: str, message: str | None) -> dict[str, str]:
    """Build a schema-valid bug report from local commit metadata.

    When the commit message is unavailable (offline, rate-limited, or a missing/invalid
    ``.gh_token`` -> 401/403), title/description would be empty. Tag the report with an
    ``error`` key so validation treats it as a non-fatal *warning* rather than hard-failing the
    whole bug over a best-effort artifact (coverage/faults are the real deliverables).
    """
    body = message or ""
    report = {
        "url": f"{github_url.rstrip('/')}/commit/{commit}",
        "title": body.splitlines()[0] if body else "",
        "description": body,
        "raw": body,
    }
    if not body:
        report["error"] = (
            "commit message unavailable (offline, rate-limited, or missing/invalid .gh_token)"
        )
    return report


def get_bug_report_bugsinpy(repo: BugsInPyRepo) -> dict[str, str] | None:
    """Source a BugsInPy bug report: fixed_commit -> commit message -> issue.

    Falls back to a minimal report (commit message / local metadata) when offline,
    rate-limited, or when the commit references no issue. Returns None only when
    the project's ``github_url`` / ``fixed_commit_id`` are unavailable.
    """
    github_url = _bip_github_url(repo.project)
    commit = repo._bug_info_value("fixed_commit_id")
    if not github_url or not commit:
        logger.info(
            "No github_url/fixed_commit_id for %s-%s; no bug report.", repo.project, repo.bug_id
        )
        return None

    owner_repo = _owner_repo(github_url)
    if owner_repo is None:
        return _minimal_bip_report(github_url, commit, None)
    owner, name = owner_repo

    message = _github_commit_message(owner, name, commit)
    if message:
        issue_match = re.search(r"#(\d+)", message)
        if issue_match:
            issue_url = f"https://github.com/{owner}/{name}/issues/{issue_match.group(1)}"
            _wait_for_domain(issue_url)
            try:
                report = parse_report(issue_url)
            except Exception as exc:
                logger.warning("Issue parse failed for %s: %s", issue_url, exc)
                report = None
            if report and "error" not in report and report.get("title"):
                return report
    return _minimal_bip_report(github_url, commit, message)


def save_bug_report(
    repo: D4JRepo | BugsInPyRepo,
    *,
    skip_existing: bool = True,
    dataset: str = "defects4j",
) -> Path | None:
    """Fetch the bug report and save as ``bug_report.json``.

    Returns the path to the written file, or None if no report is available.
    """
    output_path = repo.output_dir / "bug_report.json"

    if skip_existing and output_path.exists():
        logger.info("%s already exists, skipping.", output_path)
        return output_path

    if normalize_benchmark_name(dataset) == "BIP":
        from src.extraction.bugsinpy import BugsInPyRepo as _BIPRepo

        assert isinstance(repo, _BIPRepo)
        report = get_bug_report_bugsinpy(repo)
    else:
        assert isinstance(repo, D4JRepo)
        report = get_bug_report(repo.project, repo.bug_id)
    if report is None:
        logger.info(
            "No bug report URL for %s-%d, skipping.",
            repo.project,
            repo.bug_id,
        )
        return None

    if "error" in report:
        logger.warning(
            "Bug report fetch error for %s-%d: %s",
            repo.project,
            repo.bug_id,
            report["error"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Bug report saved to %s", output_path)
    return output_path
