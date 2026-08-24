"""Unit tests for the three-tier corpus_id fuzzy fallback in rankings.py.

Mirror-tested for both Agent4SR (``generate_top20``) and Agent4LR
(``generate_top5``).  The fallback is a strict improvement over the
prior silent drop, so previous behavior (warning + omission) is
preserved only when no tier matches.
"""

from __future__ import annotations

from src.common.method_entity import MethodEntity
from src.common.rankings import (
    _build_param_stripped_index,
    _entity_index_by_corpus_id,
    _fuzzy_match_corpus_id,
)


def _entity(corpus_id: str, *, start: int = 10, end: int = 20) -> MethodEntity:
    return MethodEntity(
        corpus_id=corpus_id,
        class_fqn_dotted=corpus_id.split("(", 1)[0].replace("$", ".").rsplit(".", 1)[0],
        path="src/X.java",
        start_line=start,
        end_line=end,
    )


def _indexes(entities: list[MethodEntity]):
    return {
        "entity_index": _entity_index_by_corpus_id(entities),
        "dotted_index": {e.corpus_id.replace("$", "."): e for e in entities},
        "stripped_index": _build_param_stripped_index(entities),
    }


def test_tier1_exact_corpus_id_match() -> None:
    entities = [_entity("org.example$Foo.do(int)")]
    matched = _fuzzy_match_corpus_id("org.example$Foo.do(int)", **_indexes(entities))
    assert matched is not None
    assert matched.corpus_id == "org.example$Foo.do(int)"


def test_tier2_dotted_form_match() -> None:
    entities = [_entity("org.example$Foo.do(int)")]
    matched = _fuzzy_match_corpus_id("org.example.Foo.do(int)", **_indexes(entities))
    assert matched is not None
    assert matched.corpus_id == "org.example$Foo.do(int)"


def test_tier3_param_stripped_fallback() -> None:
    """When parameter normalization disagrees with the corpus, the
    parameters-stripped key still matches."""
    entities = [_entity("org.example$Foo.do(java.lang.String)")]
    # Agent emits a simpler param shape that mismatches the corpus form.
    matched = _fuzzy_match_corpus_id("org.example.Foo.do(String)", **_indexes(entities))
    assert matched is not None
    assert matched.corpus_id == "org.example$Foo.do(java.lang.String)"


def test_tier3_picks_shortest_overload_deterministically() -> None:
    """On overload collisions, the fuzzy fallback picks the entity with
    the shortest (and lex-min) corpus_id for determinism."""
    entities = [
        _entity("org.example$Foo.do(java.lang.String,int)"),
        _entity("org.example$Foo.do(int)"),
        _entity("org.example$Foo.do(java.lang.Long)"),
    ]
    matched = _fuzzy_match_corpus_id("org.example.Foo.do", **_indexes(entities))
    assert matched is not None
    assert matched.corpus_id == "org.example$Foo.do(int)"


def test_returns_none_when_no_tier_matches() -> None:
    """Preserves the legacy drop-with-warning behavior at the call site."""
    entities = [_entity("org.example$Foo.do(int)")]
    matched = _fuzzy_match_corpus_id("org.other.Unrelated.method(int)", **_indexes(entities))
    assert matched is None


def test_returns_none_on_empty_corpus() -> None:
    """No entities → no matches at any tier."""
    matched = _fuzzy_match_corpus_id("org.example.Foo.do(int)", **_indexes([]))
    assert matched is None
