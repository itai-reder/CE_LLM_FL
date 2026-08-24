"""Tests for src.common.text_processor module."""

from __future__ import annotations

from pathlib import Path

from src.common.text_processor import (
    load_stopwords,
    preprocess,
    remove_stopwords,
    split_tokens,
    stem_tokens,
)

# ---------------------------------------------------------------------------
# split_tokens
# ---------------------------------------------------------------------------


class TestSplitTokens:
    """Tests for CamelCase + underscore splitting."""

    def test_camel_case(self) -> None:
        assert split_tokens("getMinValue") == ["get", "min", "value"]

    def test_upper_snake_case(self) -> None:
        assert split_tokens("MAX_VALUE") == ["max", "value"]

    def test_acronym_then_word(self) -> None:
        assert split_tokens("XMLParser") == ["xml", "parser"]

    def test_word_then_acronym_then_digit(self) -> None:
        # Regex splits on case boundaries only; digits stay attached to preceding letters
        assert split_tokens("parseHTML5") == ["parse", "html5"]

    def test_single_word(self) -> None:
        assert split_tokens("hello") == ["hello"]

    def test_empty_string(self) -> None:
        assert split_tokens("") == []

    def test_mixed_separators(self) -> None:
        result = split_tokens("get_minValue__MAX")
        assert result == ["get", "min", "value", "max"]

    def test_all_uppercase(self) -> None:
        assert split_tokens("URL") == ["url"]

    def test_numeric_suffix(self) -> None:
        assert split_tokens("value42") == ["value42"]

    def test_special_chars_stripped(self) -> None:
        result = split_tokens("foo.bar(baz)")
        assert result == ["foo", "bar", "baz"]


# ---------------------------------------------------------------------------
# load_stopwords
# ---------------------------------------------------------------------------


class TestLoadStopwords:
    """Tests for stopword loading."""

    def test_loads_bundled_stopwords(self) -> None:
        # Clear any cached result first
        load_stopwords.cache_clear()
        sw = load_stopwords()
        assert isinstance(sw, frozenset)
        assert len(sw) > 0
        # Should contain common English stopwords and Java keywords
        assert "the" in sw
        assert "public" in sw
        assert "class" in sw
        assert "return" in sw

    def test_loads_custom_file(self, tmp_path: Path) -> None:
        load_stopwords.cache_clear()
        sw_file = tmp_path / "custom_stop.txt"
        sw_file.write_text("alpha\nbeta\ngamma\n")
        sw = load_stopwords(str(sw_file))
        assert sw == frozenset({"alpha", "beta", "gamma"})
        # Clean up cache so other tests use the bundled file
        load_stopwords.cache_clear()

    def test_caching(self) -> None:
        load_stopwords.cache_clear()
        sw1 = load_stopwords()
        sw2 = load_stopwords()
        assert sw1 is sw2  # same object from cache


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------


class TestRemoveStopwords:
    """Tests for stopword removal."""

    def test_removes_stopwords(self) -> None:
        sw = frozenset({"the", "is", "a"})
        tokens = ["the", "cat", "is", "a", "big", "animal"]
        result = remove_stopwords(tokens, stopwords=sw)
        assert result == ["cat", "big", "animal"]

    def test_removes_single_char(self) -> None:
        sw: frozenset[str] = frozenset()
        tokens = ["a", "bb", "c", "dd"]
        result = remove_stopwords(tokens, stopwords=sw)
        assert result == ["bb", "dd"]

    def test_removes_pure_digits(self) -> None:
        sw: frozenset[str] = frozenset()
        tokens = ["42", "abc", "100", "x1"]
        result = remove_stopwords(tokens, stopwords=sw)
        assert result == ["abc", "x1"]

    def test_empty_input(self) -> None:
        sw: frozenset[str] = frozenset()
        assert remove_stopwords([], stopwords=sw) == []


# ---------------------------------------------------------------------------
# stem_tokens
# ---------------------------------------------------------------------------


class TestStemTokens:
    """Tests for Porter stemming."""

    def test_basic_stemming(self) -> None:
        result = stem_tokens(["running", "tests", "values"])
        assert result == ["run", "test", "valu"]

    def test_already_stemmed(self) -> None:
        result = stem_tokens(["get", "set", "add"])
        assert result == ["get", "set", "add"]

    def test_empty(self) -> None:
        assert stem_tokens([]) == []


# ---------------------------------------------------------------------------
# preprocess (full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for the full 3-stage pipeline."""

    def test_full_pipeline(self) -> None:
        sw = frozenset({"the", "is", "a", "of"})
        result = preprocess("getMinValue", stopwords=sw)
        # split: ["get", "min", "value"]
        # stop: ["get", "min", "value"] (none are stopwords or single-char)
        # stem: Porter stems
        assert len(result) == 3
        assert result[0] == "get"
        assert result[1] == "min"
        assert result[2] == "valu"  # "value" -> "valu"

    def test_pipeline_with_stopwords(self) -> None:
        sw = frozenset({"get", "set"})
        result = preprocess("getValue", stopwords=sw)
        # split: ["get", "value"]
        # stop: ["value"] (get is a stopword)
        # stem: ["valu"]
        assert result == ["valu"]

    def test_empty_string(self) -> None:
        sw: frozenset[str] = frozenset()
        assert preprocess("", stopwords=sw) == []

    def test_single_char_tokens_removed(self) -> None:
        sw: frozenset[str] = frozenset()
        # "a_B_c" -> split: ["a", "b", "c"] -> stop removes single-char -> []
        result = preprocess("a_B_c", stopwords=sw)
        assert result == []
