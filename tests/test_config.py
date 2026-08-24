"""Tests for src.common.config helpers."""

from __future__ import annotations

from src.common.config import _SLUG_MAP, _model_slug


def test_model_slug_default_replaces_colon() -> None:
    assert _model_slug("llama3.1:8b") == "llama3.1_8b"


def test_model_slug_default_no_colon_unchanged() -> None:
    assert _model_slug("mistral-7b") == "mistral-7b"


def test_model_slug_default_map_is_colon_to_underscore() -> None:
    assert _SLUG_MAP == {":": "_"}


def test_model_slug_custom_map_extends_substitutions() -> None:
    custom = {":": "_", ".": "-", "/": "__"}
    assert _model_slug("ollama/llama3.1:8b", slug_map=custom) == "ollama__llama3-1_8b"


def test_model_slug_custom_map_overrides_default() -> None:
    # Replace colon with hyphen instead of underscore.
    assert _model_slug("llama3.1:8b", slug_map={":": "-"}) == "llama3.1-8b"


def test_model_slug_empty_map_is_identity() -> None:
    assert _model_slug("llama3.1:8b", slug_map={}) == "llama3.1:8b"
