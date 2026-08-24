"""Input loading for Agent4LR.

Reads from CEFL's ``data/D4J/processed/<Project>/<BugId>/`` layout:

  - ``bug_report.json`` — ``{"title": ..., "description": ...}`` (optional)
  - ``trigger_test_clean.txt`` — combined test source + cleaned stack
    trace produced by :mod:`src.extraction.trigger_test` (optional but
    expected when ``input_keys`` contains ``"trigger_test"``)
  - ``FlexFL/SR/rankings/top20/<sr_model_id>.txt`` — numbered candidate
    list fed to LR (required)

Jira markup is stripped from the bug-report title and description at
prompt-build time only — the on-disk ``bug_report.json`` keeps the raw
text so other FL methods (BoostN/SBIR) see unchanged input.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import get_lr_candidate_file, get_processed_dir

_JIRA_MONOSPACE_RE = re.compile(r"\{\{(.+?)\}\}", flags=re.DOTALL)
_JIRA_BLOCK_TAG_RE = re.compile(r"\{(quote|noformat|code|panel)(?::[^}]*)?\}")
# Match *emphasis* / _underline_ only when adjacent to whitespace/punctuation/EOS,
# never inside an identifier. The negative lookarounds (?<!\w) / (?!\w) keep
# `String.equals` / `__init__` untouched.
_JIRA_EMPHASIS_RE = re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", flags=re.DOTALL)
_JIRA_UNDERLINE_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", flags=re.DOTALL)
_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_jira(text: str) -> str:
    """Remove Jira-flavoured markup from a bug-report text.

    Targets the subset that actually shows up in the issue-tracker dumps
    we feed to LR: monospace ``{{x}}`` wrappers, block tags ``{quote}``
    / ``{code}`` / ``{noformat}`` / ``{panel}`` (with optional
    ``:lang``-style suffixes), inline emphasis / underline, and HTML
    entities (``&lt;``, ``&gt;``, ``&amp;``, ``&quot;``). Finally
    collapses runs of 3+ blank lines back to a single blank line — the
    block-tag deletions tend to leave gaps.
    """
    text = _JIRA_MONOSPACE_RE.sub(r"\1", text)
    text = _JIRA_BLOCK_TAG_RE.sub("", text)
    text = _JIRA_EMPHASIS_RE.sub(r"\1", text)
    text = _JIRA_UNDERLINE_RE.sub(r"\1", text)
    text = html.unescape(text)
    text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
    return text


@dataclass(frozen=True)
class LRBugInputs:
    """Inputs required by the Agent4LR pipeline for a single bug.

    ``candidates`` is the top-20 list of FQNs in display form (dotted
    package + simple param types), as written by
    :func:`src.common.rankings.generate_top20`. Position in the list is
    the 1-based ``method_number`` argument used by the structured
    ``get_snippet_of_method`` tool.

    ``trigger_test_clean`` is the **single combined** text block (test
    method source truncated at the failing assertion + transition phrase
    + cleaned stack trace), not the raw ``failing_tests`` file.
    """

    project: str
    bug_id: str
    sr_model_id: str
    candidates: list[str]
    bug_report_title: str | None
    bug_report_description: str | None
    trigger_test_clean: str | None

    def has_bug_report(self) -> bool:
        """True iff a bug report title or description is available."""
        return bool(self.bug_report_title) or bool(self.bug_report_description)

    def has_trigger_test(self) -> bool:
        """True iff a non-empty cleaned trigger test is available."""
        return bool(self.trigger_test_clean)

    def bug_report_text_block(self) -> str | None:
        """Render the bug report as a planner-prompt text block.

        Strips Jira markup from both title and description, then wraps
        them in the reference pipeline's verbatim framing::

            The bug report is as follows:
            ```
            Title:<stripped title>
            Description:<stripped description>```

        Note the absence of newlines between ``Description:`` and the
        body, and between the body and the closing fence — the
        description's own leading newlines (if any) provide visual
        separation, matching FlexFL's ``get_input_data``.
        """
        if not self.has_bug_report():
            return None
        title = _strip_jira(self.bug_report_title or "")
        desc = _strip_jira(self.bug_report_description or "")
        return f"The bug report is as follows:\n```\nTitle:{title}\nDescription:{desc}```"

    def trigger_test_text_block(self) -> str | None:
        """Format the cleaned trigger test as a text block for the LLM prompt."""
        if not self.has_trigger_test():
            return None
        return f"The triggering test traceback is as follows:\n```\n{self.trigger_test_clean}\n```"

    def candidate_list_text_block(self) -> str:
        """Render the candidate list as a numbered block (1-based)."""
        lines = [f"{i}. {fqn}" for i, fqn in enumerate(self.candidates, start=1)]
        body = "\n".join(lines)
        return f"The most likely culprit methods are:\n```\n{body}\n```"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_lr_bug_inputs(
    project: str,
    bug_id: str | int,
    *,
    sr_model_id: str,
    input_keys: tuple[str, ...] = ("bug_report", "trigger_test"),
    dataset: str = "defects4j",
) -> LRBugInputs:
    """Load LR inputs for one bug.

    Reads (under the processed dir):

      * ``bug_report.json`` when ``"bug_report" in input_keys``
      * ``trigger_test_clean.txt`` when ``"trigger_test" in input_keys``
      * ``FlexFL/SR/rankings/top20/<sr_model_id>.txt`` (required)

    Returns the populated :class:`LRBugInputs`. Inputs absent from disk
    but requested in ``input_keys`` are simply ``None``/empty in the
    returned dataclass — the runner records the actual intersection.

    Raises
    ------
    FileNotFoundError
        If the candidate-list file is missing.
    """
    processed = get_processed_dir(project, bug_id, dataset=dataset)
    bug_id_str = str(bug_id)

    cand_path = get_lr_candidate_file(project, bug_id, sr_model_id=sr_model_id, dataset=dataset)
    if not cand_path.exists():
        raise FileNotFoundError(
            f"Candidate list not found at {cand_path}. "
            f"Run the SR pipeline for {project}-{bug_id} with --model whose "
            f"slug matches sr_model_id={sr_model_id!r} first."
        )
    candidates = [line for line in _read_text(cand_path).splitlines() if line.strip()]

    title: str | None = None
    desc: str | None = None
    if "bug_report" in input_keys:
        br_path = processed / "bug_report.json"
        if br_path.exists():
            raw = _read_json(br_path)
            if isinstance(raw, dict):
                title_val = raw.get("title")
                desc_val = raw.get("description")
                title = title_val if isinstance(title_val, str) else None
                desc = desc_val if isinstance(desc_val, str) else None

    trigger: str | None = None
    if "trigger_test" in input_keys:
        tt_path = processed / "trigger_test_clean.txt"
        if tt_path.exists():
            trigger = _read_text(tt_path)

    return LRBugInputs(
        project=project,
        bug_id=bug_id_str,
        sr_model_id=sr_model_id,
        candidates=candidates,
        bug_report_title=title,
        bug_report_description=desc,
        trigger_test_clean=trigger,
    )
