"""Traditional FL on BugsInPy.

Covers: the SBFL Python-shape statement-ID reduction,
the benchmark-aware parser/corpus dispatch (``source_corpus``), and the
dataset-aware stopwords selection. End-to-end runs against real bugs are exercised
by the on-disk verification, not here.
"""

from __future__ import annotations

from pathlib import Path

from src.common import config
from src.common.source_corpus import iter_method_corpus, iter_statement_corpus
from src.sbir.sbfl import SBFL


class TestSbflPythonReduction:
    """``parse_ochiai_csv`` on a FauxPy (no-``#``) ranking reduces to class-bearing IDs."""

    def test_python_shape_ranking_non_empty(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text(
            "name;suspiciousness_value\n"
            "youtube_dl.utils$parse_duration(s):1828;0.9\n"
            "youtube_dl.extractor$DASHIE.real_extract(self,url):42;0.5\n"
        )
        scores = SBFL.parse_ochiai_csv(csv_file, dataset="bugsinpy")
        assert scores, "Python-shape SBFL reduction dropped all rows (owner-reduction regression)"
        # module-level function -> the module owns the line (no '#'-style class part)
        assert scores["youtube_dl.utils#1828"] == 0.9
        # class method -> class-bearing fqn
        assert scores["youtube_dl.extractor.DASHIE#42"] == 0.5

    def test_d4j_shape_unchanged(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text("name;suspiciousness_value\norg.example$Foo#bar():10;0.75\n")
        # default dataset is defects4j; the Java behavior is unchanged
        assert SBFL.parse_ochiai_csv(csv_file) == {"org.example.Foo#10": 0.75}


class TestStopwordsSelection:
    def test_python_for_bugsinpy(self) -> None:
        assert config.get_stopwords_file("bugsinpy") == config.STOPWORDS_FILE_PYTHON
        assert config.get_stopwords_file("bip") == config.STOPWORDS_FILE_PYTHON

    def test_java_for_defects4j(self) -> None:
        assert config.get_stopwords_file("defects4j") == config.STOPWORDS_FILE
        assert config.get_stopwords_file() == config.STOPWORDS_FILE

    def test_python_stopwords_has_keywords_not_java(self) -> None:
        words = {
            w.strip().lower()
            for w in config.STOPWORDS_FILE_PYTHON.read_text(encoding="utf-8").splitlines()
            if w.strip()
        }
        assert {"def", "lambda", "self", "cls", "yield"} <= words
        # Java-only keywords must not leak into the Python list
        assert "instanceof" not in words
        assert "synchronized" not in words


_PY_SOURCE = """\
import os


def top_level(a, b):
    x = a + b
    return x


class Greeter:
    def greet(self, name):
        message = "hi " + name
        return message
"""


class TestSourceCorpusDispatch:
    """``iter_method_corpus`` / ``iter_statement_corpus`` produce the locked Python shapes."""

    @staticmethod
    def _write_source(tmp_path: Path) -> Path:
        (tmp_path / "mypkg").mkdir()
        (tmp_path / "mypkg" / "mod.py").write_text(_PY_SOURCE, encoding="utf-8")
        return tmp_path  # the import root / source_root

    def test_method_corpus_python_ids(self, tmp_path: Path) -> None:
        root = self._write_source(tmp_path)
        pairs = iter_method_corpus(root, "bugsinpy")
        ids = {corpus_id for _, corpus_id in pairs}
        assert "mypkg.mod$top_level(a,b)" in ids
        assert "mypkg.mod$Greeter.greet(self,name)" in ids
        # MethodInfo.content is carried through for BM25 scoring
        assert all(method_info.content for method_info, _ in pairs)

    def test_statement_corpus_python_ids(self, tmp_path: Path) -> None:
        root = self._write_source(tmp_path)
        stmts = iter_statement_corpus(root, "bugsinpy")
        ids = {s.stmt_id for s in stmts}
        # class-bearing owner for class-body lines
        assert any(i.startswith("mypkg.mod.Greeter#") for i in ids)
        # module-level owner for top-level lines
        assert any(i.startswith("mypkg.mod#") and "Greeter" not in i for i in ids)

    def test_dispatch_matches_java_branch_shape(self, tmp_path: Path) -> None:
        """The D4J branch still routes to the Java parser (no Python ids)."""
        (tmp_path / "Foo.java").write_text(
            "package p;\npublic class Foo {\n  int bar() { return 1; }\n}\n",
            encoding="utf-8",
        )
        pairs = iter_method_corpus(tmp_path, "defects4j")
        # Java corpus ids use the '$' package/class separator, never a module-dotted form
        assert all("$" in corpus_id for _, corpus_id in pairs)
