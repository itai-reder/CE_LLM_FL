"""Tests for src.common.method_entity — corpus-ID → owner recovery and the
statement→method aggregation join.

Regression anchor for the owner-recovery rule: the join is an exact
``(class_fqn_dotted, line)`` dict lookup, where ``class_fqn_dotted`` is recovered
from the corpus_id by :func:`_class_fqn_from_corpus_id`.  For a statement score to
reach its method, the statement-ID ``text-before-#`` must equal that recovered owner.
These tests lock the recovery for both Java (no regression) and Python (class-bearing
statement IDs; module-level functions owned by the module) shapes.
"""

from __future__ import annotations

import pytest

from src.common.method_entity import (
    MethodEntity,
    _class_fqn_from_corpus_id,
    aggregate_statement_scores_to_entities,
    build_entity_line_index,
)


class TestClassFqnFromCorpusId:
    """Recover the statement-owner FQN from a corpus identity string."""

    # Java shapes — these MUST stay byte-identical so the shared D4J
    # aggregation does not regress.
    @pytest.mark.parametrize(
        ("corpus_id", "expected"),
        [
            ("org.example$SampleClass.add(int,int)", "org.example.SampleClass"),
            (
                "org.apache.commons.lang3$StringUtils.isEmpty(CharSequence)",
                "org.apache.commons.lang3.StringUtils",
            ),
            # inner class — '$' before the simple class name flattens to '.'
            ("org.example.Outer$Inner.method()", "org.example.Outer.Inner"),
            # inner class — '$' before package/class split, second '$' inside qualname
            ("org.example$Outer$Inner.method()", "org.example.Outer.Inner"),
            # default package (no '$'): owner is the substring before the last '.'
            ("Foo.bar()", "Foo"),
        ],
    )
    def test_java_shapes_unchanged(self, corpus_id: str, expected: str) -> None:
        assert _class_fqn_from_corpus_id(corpus_id) == expected

    # Python shapes.
    @pytest.mark.parametrize(
        ("corpus_id", "expected"),
        [
            # class method: owner is module + dotted class
            (
                "src.common.method_entity$MethodEntity.__hash__(self)",
                "src.common.method_entity.MethodEntity",
            ),
            # nested class
            ("mod$Outer.Inner.m(self)", "mod.Outer.Inner"),
            # module-level function: the MODULE owns the line.
            # The Java-shaped recovery used to amputate a module segment here
            # (returning "src.common"), silently dropping every module-func line.
            (
                "src.common.method_entity$method_info_to_corpus_id(m)",
                "src.common.method_entity",
            ),
            # top-level module (no package) function
            ("foo$bar(s)", "foo"),
        ],
    )
    def test_python_shapes(self, corpus_id: str, expected: str) -> None:
        assert _class_fqn_from_corpus_id(corpus_id) == expected


class TestPythonStatementAggregation:
    """End-to-end: a class-bearing Python statement key reaches its method entity.

    Mirrors the worked example on src/common/method_entity.py itself:
    a class method (``MethodEntity.__hash__``) and a module-level function
    (``method_info_to_corpus_id``).
    """

    MODULE = "src.common.method_entity"
    CLASS_METHOD_ID = f"{MODULE}$MethodEntity.__hash__(self)"
    MODULE_FUNC_ID = f"{MODULE}$method_info_to_corpus_id(m)"

    def _entities(self) -> list[MethodEntity]:
        # class_fqn_dotted is recovered from corpus_id exactly as
        # load_method_entities() does on reload.
        return [
            MethodEntity(
                corpus_id=self.CLASS_METHOD_ID,
                class_fqn_dotted=_class_fqn_from_corpus_id(self.CLASS_METHOD_ID),
                path="src/common/method_entity.py",
                start_line=38,
                end_line=44,
            ),
            MethodEntity(
                corpus_id=self.MODULE_FUNC_ID,
                class_fqn_dotted=_class_fqn_from_corpus_id(self.MODULE_FUNC_ID),
                path="src/common/method_entity.py",
                start_line=47,
                end_line=69,
            ),
        ]

    def test_class_method_line_reaches_its_method(self) -> None:
        entities = self._entities()
        index = build_entity_line_index(entities)
        # statement ID "src.common.method_entity.MethodEntity#39"
        stmt_key = (f"{self.MODULE}.MethodEntity", 39)
        scores = aggregate_statement_scores_to_entities({stmt_key: 0.9}, index)
        assert scores == {entities[0]: 0.9}

    def test_module_function_line_reaches_its_method(self) -> None:
        entities = self._entities()
        index = build_entity_line_index(entities)
        # statement ID "src.common.method_entity#56" (module owns the line)
        stmt_key = (self.MODULE, 56)
        scores = aggregate_statement_scores_to_entities({stmt_key: 0.7}, index)
        assert scores == {entities[1]: 0.7}

    def test_class_less_statement_key_is_dropped(self) -> None:
        """The old class-less form for a class-method line joins to nothing."""
        entities = self._entities()
        index = build_entity_line_index(entities)
        # Old (broken) convention dropped the class: "src.common.method_entity#39".
        # Line 39 is inside the class method's range but the key lacks the class,
        # so the exact-tuple lookup misses and the score is silently dropped.
        stmt_key = (self.MODULE, 39)
        scores = aggregate_statement_scores_to_entities({stmt_key: 0.9}, index)
        assert scores == {}
