"""Bug report parser for all Defects4J URL types.

Supports:
  - SourceForge bugs/patches  (Chart, Time)
  - Apache JIRA REST API      (Cli, Codec, Collections, Compress, Csv, JxPath, Lang, Math)
  - Google Code Archive JSON   (Closure)
  - GitHub issues              (Gson, JacksonCore, JacksonDatabind, JacksonXml, Jsoup, Mockito)
  - GitHub pull requests       (Gson, JacksonCore, JacksonDatabind, Mockito)
  - code.google.com/archive    (Mockito older bugs → Google Code Archive JSON)
"""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

from src.common.github_token import get_github_token


def parse_report(url: str) -> dict[str, str]:
    """Parse a Defects4J bug report URL and return title + description.

    Returns a dict with keys ``url``, ``title``, ``raw``, ``description``.
    On failure the dict contains ``url`` and ``error`` instead.
    """
    d: dict[str, str] = {"url": url}
    if url == "UNKNOWN" or not url:
        d["error"] = "URL is UNKNOWN or empty"
        return d

    try:
        if "sourceforge.net/p/" in url:
            parsed = _parse_sourceforge(url)
        elif "issues.apache.org/jira/browse/" in url:
            parsed = _parse_apache_jira(url)
        elif "storage.googleapis.com/google-code-archive/" in url:
            parsed = _parse_google_code_archive_json(url)
        elif "code.google.com/archive/p/" in url:
            parsed = _parse_code_google_com_archive(url)
        elif "github.com/" in url and "/pull/" in url:
            parsed = _parse_github_pull(url)
        elif "github.com/" in url and "/issues/" in url:
            parsed = _parse_github_issue(url)
        else:
            d["error"] = f"Unrecognized URL type: {url}"
            return d
        d.update(parsed)
    except Exception as exc:
        d["error"] = f"Failed to parse report: {exc}"

    return d


# ---------------------------------------------------------------------------
# SourceForge (bugs and patches)
# ---------------------------------------------------------------------------


def _parse_sourceforge(url: str) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return {"error": f"Failed to fetch page. Status code: {response.status_code}"}

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h2", class_="dark title")
    title = " ".join(title_tag.get_text().split()) if title_tag else "Title not found"

    desc_div = soup.find("div", class_="markdown_content")
    if desc_div:
        raw = desc_div.get_text()
        description = _plaintext_to_readable(raw)
    else:
        return {"error": "Description div not found in SourceForge page"}

    return {"title": title, "raw": raw, "description": description}


# ---------------------------------------------------------------------------
# Apache JIRA (REST API)
# ---------------------------------------------------------------------------


def _parse_apache_jira(url: str) -> dict[str, str]:
    match = re.search(r"/browse/([A-Z]+-\d+)", url)
    if not match:
        return {"error": f"Could not extract JIRA issue key from URL: {url}"}

    issue_key = match.group(1)
    api_url = (
        f"https://issues.apache.org/jira/rest/api/2/issue/{issue_key}?fields=summary,description"
    )

    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        return {"error": f"JIRA API returned status {response.status_code}"}

    data = response.json()
    fields = data.get("fields", {})

    title = fields.get("summary", "Title not found")
    raw = fields.get("description", "Description not found") or "Description not found"
    description = _plaintext_to_readable(raw)

    return {"title": title, "raw": raw, "description": description}


# ---------------------------------------------------------------------------
# Google Code Archive JSON
# ---------------------------------------------------------------------------


def _parse_google_code_archive_json(url: str) -> dict[str, str]:
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return {"error": f"Google Code Archive returned status {response.status_code}"}

    data = response.json()
    title = data.get("summary", "Title not found")

    comments = data.get("comments", [])
    raw = (
        comments[0].get("content", "Description not found") if comments else "Description not found"
    )
    description = _html_to_readable(raw)

    return {"title": title, "raw": raw, "description": description}


# ---------------------------------------------------------------------------
# code.google.com/archive → rewrite to Google Code Archive JSON
# ---------------------------------------------------------------------------


def _parse_code_google_com_archive(url: str) -> dict[str, str]:
    match = re.search(r"code\.google\.com/archive/p/([^/]+)/issues/(\d+)", url)
    if not match:
        return {"error": f"Could not parse code.google.com archive URL: {url}"}

    project = match.group(1)
    issue_num = match.group(2)
    json_url = (
        "https://storage.googleapis.com/google-code-archive/v2/"
        f"code.google.com/{project}/issues/issue-{issue_num}.json"
    )
    return _parse_google_code_archive_json(json_url)


# ---------------------------------------------------------------------------
# GitHub Issues (REST API)
# ---------------------------------------------------------------------------


def _parse_github_issue(url: str) -> dict[str, str]:
    match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if not match:
        return {"error": f"Could not parse GitHub issue URL: {url}"}

    owner, repo, number = match.group(1), match.group(2), match.group(3)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    return _fetch_github_api(api_url)


# ---------------------------------------------------------------------------
# GitHub Pull Requests (REST API)
# ---------------------------------------------------------------------------


def _parse_github_pull(url: str) -> dict[str, str]:
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not match:
        return {"error": f"Could not parse GitHub PR URL: {url}"}

    owner, repo, number = match.group(1), match.group(2), match.group(3)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    return _fetch_github_api(api_url)


def _fetch_github_api(api_url: str) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = get_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(api_url, headers=headers, timeout=30)
    if response.status_code != 200:
        return {"error": f"GitHub API returned status {response.status_code}"}

    data = response.json()
    title = data.get("title", "Title not found")
    raw = data.get("body", "Description not found") or "Description not found"
    description = _plaintext_to_readable(raw)

    return {"title": title, "raw": raw, "description": description}


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_CONTINUATION_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "to",
    "for",
    "and",
    "or",
    "but",
    "with",
    "by",
    "as",
    "at",
    "on",
    "from",
    "into",
    "that",
    "which",
    "who",
    "whom",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "not",
    "no",
    "this",
    "these",
    "than",
    "if",
    "so",
    "yet",
    "nor",
}

_STOP_JOIN_CHARS = re.compile(r"[.!?;{}\]]$")


def _should_join(prev: str, cur_line: str) -> bool:
    if not prev:
        return False
    if cur_line and cur_line[0].isspace():
        return False

    stripped_prev = prev.strip()
    stripped_cur = cur_line.strip()
    if not stripped_cur:
        return False

    last_word = stripped_prev.rsplit(None, 1)[-1].lower().rstrip(",;:")
    if last_word in _CONTINUATION_WORDS:
        return True
    if _STOP_JOIN_CHARS.search(stripped_prev):
        return False
    return bool(stripped_cur[0].islower())


def _html_to_readable(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    return _plaintext_to_readable(soup.get_text())


def _plaintext_to_readable(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(```(?:.|\n)*?```)", text)

    processed_parts: list[str] = []
    for part in parts:
        if part.startswith("```"):
            processed_parts.append(part.strip())
        else:
            processed_parts.append(_process_prose(part))

    return "\n\n".join(p for p in processed_parts if p)


def _process_prose(text: str) -> str:
    blocks = re.split(r"\n\n+", text)
    cleaned_blocks: list[str] = []

    for block in blocks:
        if not block.strip():
            continue

        lines = block.split("\n")
        if not lines:
            continue

        merged = [lines[0].rstrip()]

        for line in lines[1:]:
            prev = merged[-1]
            stripped_prev = prev.rstrip()
            stripped_cur = line.strip()

            if stripped_prev.endswith(":") or (
                stripped_prev.endswith((";", "}")) and stripped_cur and stripped_cur[0].isupper()
            ):
                merged.append("")
                merged.append(line.rstrip())
            elif _should_join(prev, line):
                merged[-1] = prev + " " + line.strip()
            else:
                merged.append(line.rstrip())

        cleaned_blocks.append("\n".join(merged))

    return "\n\n".join(cleaned_blocks)
