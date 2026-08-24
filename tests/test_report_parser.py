"""Tests for src.extraction.report_parser — text cleaning helpers."""

from __future__ import annotations

from src.extraction.report_parser import (
    _html_to_readable,
    _plaintext_to_readable,
    _should_join,
    parse_report,
)


class TestShouldJoin:
    """Tests for _should_join()."""

    def test_continuation_word(self) -> None:
        assert _should_join("This is a", "continuation") is True

    def test_stop_char(self) -> None:
        assert _should_join("End of sentence.", "Next sentence") is False

    def test_lowercase_start(self) -> None:
        assert _should_join("some text", "continues here") is True

    def test_uppercase_start(self) -> None:
        assert _should_join("some text", "Next paragraph") is False

    def test_empty_prev(self) -> None:
        assert _should_join("", "some text") is False

    def test_indented_cur(self) -> None:
        assert _should_join("before", "  indented") is False

    def test_empty_cur(self) -> None:
        assert _should_join("prev", "") is False

    def test_semicolon_stop(self) -> None:
        assert _should_join("x = 1;", "y = 2") is False

    def test_bracket_stop(self) -> None:
        assert _should_join("return x}", "public void") is False


class TestPlaintextToReadable:
    """Tests for _plaintext_to_readable()."""

    def test_simple_paragraph(self) -> None:
        text = "Hello world.\nThis is a test."
        result = _plaintext_to_readable(text)
        assert "Hello world." in result
        assert "This is a test." in result

    def test_code_block_preserved(self) -> None:
        text = "Before.\n```\ncode line\n```\nAfter."
        result = _plaintext_to_readable(text)
        assert "```\ncode line\n```" in result

    def test_multiple_paragraphs(self) -> None:
        text = "Para one.\n\nPara two."
        result = _plaintext_to_readable(text)
        assert "Para one." in result
        assert "Para two." in result

    def test_crlf_normalised(self) -> None:
        text = "Line one.\r\nLine two."
        result = _plaintext_to_readable(text)
        assert "\r" not in result

    def test_continuation_joined(self) -> None:
        text = "This is a\ncontinuation of the\nsentence."
        result = _plaintext_to_readable(text)
        assert "This is a continuation of the" in result


class TestHtmlToReadable:
    """Tests for _html_to_readable()."""

    def test_strips_tags(self) -> None:
        html = "<p>Hello <b>world</b></p>"
        result = _html_to_readable(html)
        assert "Hello world" in result
        assert "<p>" not in result
        assert "<b>" not in result


class TestParseReport:
    """Tests for parse_report() — edge cases only (no network)."""

    def test_unknown_url(self) -> None:
        result = parse_report("UNKNOWN")
        assert "error" in result

    def test_empty_url(self) -> None:
        result = parse_report("")
        assert "error" in result

    def test_unrecognised_url(self) -> None:
        result = parse_report("https://example.com/some/random/page")
        assert "error" in result
        assert "Unrecognized" in result["error"]
