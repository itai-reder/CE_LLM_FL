"""Input loading for Agent4SR: trigger tests and bug reports.

Reads from CEFL's ``data/D4J/processed/<Project>/<BugId>/`` layout:
  - ``trigger_test_clean.txt`` — cleaned failing-test context (test source
    plus a trimmed, app-relevant stack trace); falls back to the raw
    ``trigger_tests`` dump (test-name header + full untrimmed trace).
  - ``bug_report.json`` — ``{"title": ..., "description": ...}``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import get_processed_dir


@dataclass(frozen=True)
class BugInputs:
    """Inputs required by the Agent4SR pipeline for a single bug."""

    project: str
    bug_id: str
    trigger_test: str
    bug_report_title: str | None
    bug_report_description: str | None

    def bug_report_text_block(self) -> str | None:
        """Format the bug report as a text block for the LLM prompt.

        Returns None if no bug report is available.
        """
        if not self.bug_report_title and not self.bug_report_description:
            return None
        title = self.bug_report_title or ""
        desc = self.bug_report_description or ""
        return f"The bug report is as follows:\n```\nTitle:{title}\nDescription:\n{desc}\n```"

    def trigger_test_text_block(self) -> str:
        """Format the trigger test as a text block for the LLM prompt."""
        return f"The trigger test is as follows:\n```\n{self.trigger_test}\n```"


def _read_text(path: Path) -> str:
    """Read a text file, stripping trailing newline."""
    return path.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_bug_inputs(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> BugInputs:
    """Load trigger test and bug report for a given bug.

    Reads from ``data/D4J/processed/<Project>/<BugId>/``:
      - ``trigger_test_clean.txt`` — cleaned failing-test context (multi-line:
        test source + trimmed stack trace), with the raw ``trigger_tests``
        dump as a fallback when the clean artifact is absent
      - ``bug_report.json`` — optional JSON with ``title`` and ``description``

    Parameters
    ----------
    project:
        Defects4J project identifier (e.g. ``"Chart"``).
    bug_id:
        Bug/version number.

    Returns
    -------
    BugInputs
        Dataclass with trigger test text and optional bug report fields.
        ``trigger_test`` is empty when neither the cleaned artifact nor the
        raw dump is present.
    """
    processed = get_processed_dir(project, bug_id, dataset=dataset)
    bug_id_str = str(bug_id)

    # Trigger test — prefer the cleaned artifact, fall back to the raw dump,
    # else empty (some bugs may lack any trigger context; degrade gracefully).
    trigger_test = ""
    clean_path = processed / "trigger_test_clean.txt"
    raw_path = processed / "trigger_tests"
    if clean_path.exists():
        trigger_test = _read_text(clean_path)
    elif raw_path.exists():
        trigger_test = _read_text(raw_path)

    # Bug report — optional
    bug_report_path = processed / "bug_report.json"
    title: str | None = None
    desc: str | None = None
    if bug_report_path.exists():
        raw = _read_json(bug_report_path)
        if isinstance(raw, dict):
            title_val = raw.get("title")
            desc_val = raw.get("description")
            title = title_val if isinstance(title_val, str) else None
            desc = desc_val if isinstance(desc_val, str) else None

    return BugInputs(
        project=project,
        bug_id=bug_id_str,
        trigger_test=trigger_test,
        bug_report_title=title,
        bug_report_description=desc,
    )
