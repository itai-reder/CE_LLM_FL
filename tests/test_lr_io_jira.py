"""Tests for the Jira-markup stripper in ``src.agent4lr.io``.

Covers each markup form we expect in real bug reports, plus a
no-markup pass-through, plus an integration check that pipes a
realistic Jira-flavoured description through ``bug_report_text_block``.
"""

from __future__ import annotations

from src.agent4lr.io import LRBugInputs, _strip_jira


def test_strip_monospace_braces() -> None:
    assert _strip_jira("Hello {{world}}!") == "Hello world!"


def test_strip_quote_block_tags() -> None:
    text = "Before\n{quote}\ninner\n{quote}\nAfter"
    out = _strip_jira(text)
    assert "{quote}" not in out
    assert "inner" in out
    assert "Before" in out and "After" in out


def test_strip_code_block_with_language_suffix() -> None:
    text = "x\n{code:java}\nint y;\n{code}\nz"
    out = _strip_jira(text)
    assert "{code" not in out
    assert "int y;" in out


def test_strip_noformat_and_panel_tags() -> None:
    text = "a {noformat}b{noformat} c {panel:title=Foo}d{panel} e"
    out = _strip_jira(text)
    assert "{noformat" not in out
    assert "{panel" not in out
    assert "b" in out and "d" in out


def test_strip_emphasis_and_underline_when_word_bounded() -> None:
    assert _strip_jira("see *emphasised text* here") == "see emphasised text here"
    assert _strip_jira("see _underlined_ here") == "see underlined here"


def test_emphasis_leaves_identifiers_alone() -> None:
    # *Var* inside an identifier-like context must not be eaten.
    assert _strip_jira("a*b*c") == "a*b*c"
    # Same for _underscores_ that are part of an identifier.
    assert _strip_jira("foo_bar_baz") == "foo_bar_baz"


def test_html_entities_decoded() -> None:
    assert _strip_jira("1 &lt; 2 &amp;&amp; 3 &gt; 1") == "1 < 2 && 3 > 1"
    assert _strip_jira("a &quot;b&quot; c") == 'a "b" c'


def test_no_markup_passthrough() -> None:
    text = "Plain prose with no markup. No braces, no entities."
    assert _strip_jira(text) == text


def test_blank_line_runs_collapsed_to_two() -> None:
    assert _strip_jira("a\n\n\n\nb") == "a\n\nb"


def test_bug_report_text_block_strips_markup_in_both_fields() -> None:
    bug = LRBugInputs(
        project="P",
        bug_id="1",
        sr_model_id="m",
        candidates=["a"],
        bug_report_title="LookupTranslator accepts {{CharSequence}}",
        bug_report_description=(
            "The core of {{org.apache.commons.lang3.text.translate}} is broken.\n"
            "\n"
            "{quote}\nA char buffer is not equal to any other type of object.\n"
            "{quote}\n"
            "\n"
            "Example:\n"
            "{code:java}\n"
            'CharSequence cs1 = "1 &lt; 2";\n'
            "{code}\n"
        ),
        trigger_test_clean=None,
    )
    block = bug.bug_report_text_block()
    assert block is not None
    # framing matches reference verbatim
    assert block.startswith("The bug report is as follows:\n```\nTitle:")
    assert block.endswith("```")
    # markup removed
    assert "{{" not in block
    assert "{quote}" not in block
    assert "{code" not in block
    assert "&lt;" not in block
    # the content survives
    assert "CharSequence" in block
    assert "org.apache.commons.lang3.text.translate" in block
    assert '"1 < 2"' in block
