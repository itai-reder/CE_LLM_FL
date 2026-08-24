"""BugsInPy corpus generation + Python-ID retrieval-layer tests.

Covers:
  * the Python corpus-id round-trip through ``generate_corpus`` /
    ``iter_method_corpus`` (class method, module-level function, nested class);
  * ``function_call`` Python-ID edge cases — the module-level-function = empty
    class contract (no blank class ever surfaces; the module owns the function);
  * the hard valid-bug gate in ``run_agent4sr.cmd_corpus_bugsinpy`` (a bug whose
    extraction did not validate is logged-and-skipped, never given an empty corpus).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent4sr import function_call as fc
from src.agent4sr.corpus import Corpus, generate_corpus, save_corpus

# A module-level function, a class method, and a nested-class method — the three
# Python shapes the corpus-id construction must get right (module "pkg.mod").
PY_SOURCE = """\
def top_level(a, b):
    return a + b


class Outer:
    def method(self, x):
        return x

    class Inner:
        def inner_method(self, y):
            return y
"""


def _write_py_src(tmp_path: Path) -> Path:
    src_dir = tmp_path / "src"
    (src_dir / "pkg").mkdir(parents=True)
    (src_dir / "pkg" / "mod.py").write_text(PY_SOURCE, encoding="utf-8")
    return src_dir


class TestPythonCorpusIdRoundTrip:
    """generate_corpus(dataset="bugsinpy") emits the locked Python corpus-id shapes."""

    def test_corpus_ids(self, tmp_path: Path) -> None:
        src_dir = _write_py_src(tmp_path)
        with patch("src.agent4sr.corpus.get_src_dir", return_value=src_dir):
            corpus = generate_corpus("PySnooper", "3", dataset="bugsinpy")

        ids = set(corpus.method_ids)
        assert "pkg.mod$top_level(a,b)" in ids  # module-level function: no class
        assert "pkg.mod$Outer.method(self,x)" in ids  # class method
        assert "pkg.mod$Outer.Inner.inner_method(self,y)" in ids  # nested class
        # Parallel, non-empty, every id is a Python id (has exactly one '$').
        assert corpus.method_ids and len(corpus.method_ids) == len(corpus.raw_codes)
        assert all(e.count("$") == 1 for e in corpus.method_ids)
        # Raw code preserves real source (multi-line, indented) — not space-flattened.
        idx = corpus.method_ids.index("pkg.mod$Outer.method(self,x)")
        assert corpus.raw_codes[idx] == "    def method(self, x):\n        return x"
        ast.parse("\n".join(line[4:] for line in corpus.raw_codes[idx].splitlines()))

    def test_save_load_roundtrip_multiline(self, tmp_path: Path, monkeypatch) -> None:
        """save_corpus → load_corpus_codes round-trips a multi-line method losslessly,
        and the on-disk file keeps exactly one physical line per method (JSON-encoded)."""
        import src.agent4sr.corpus as corpus_mod

        sr = tmp_path / "SR"
        sr.mkdir()
        code = "def f(x):\n    if x:\n        return 1\n    return 0"
        monkeypatch.setattr(corpus_mod, "get_sr_dir", lambda *a, **k: sr)
        monkeypatch.setattr(
            corpus_mod,
            "generate_corpus",
            lambda p, b, *, dataset: Corpus(method_ids=["pkg.mod$f(x)"], raw_codes=[code]),
        )
        save_corpus("P", 1, dataset="bugsinpy", skip_existing=False)

        # One method → exactly one physical line on disk (+ trailing newline); no leak.
        on_disk = (sr / "corpus_codes.txt").read_text(encoding="utf-8")
        assert on_disk.count("\n") == 1

        with patch.object(fc, "get_sr_dir", return_value=sr):
            loaded = fc.load_corpus_codes("P", 1, dataset="bugsinpy")
        assert loaded == [code]
        ast.parse(loaded[0])  # faithful, parseable Python


# A synthetic Python corpus mixing class methods, a nested class, and module-level
# functions (incl. a module that has ONLY module-level functions).
PY_CORPUS_METHODS = [
    "pkg.mod$top_level(a,b)",  # module-level function (owner == module)
    "pkg.mod$Outer.method(self,x)",  # class method
    "pkg.mod$Outer.Inner.inner_method(self,y)",  # nested class method
    "pkg.only_funcs$helper(z)",  # a module with ONLY module-level functions
]
# Codes are multi-line (real Python indentation) to exercise the JSON-per-line encoding.
PY_CORPUS_CODES = [
    "def top_level(a, b):\n    return a + b",
    "def method(self, x):\n    return x",
    "def inner_method(self, y):\n    return y",
    "def helper(z):\n    if z:\n        return z\n    return None",
]


class TestFunctionCallPythonIds:
    """The 7 retrieval functions honor the module-level-function contract."""

    def _corpus(self, tmp_path: Path) -> Path:
        d = tmp_path / "SR"
        d.mkdir(parents=True)
        (d / "corpus_methods.txt").write_text("\n".join(PY_CORPUS_METHODS) + "\n", encoding="utf-8")
        # BugsInPy codes are JSON-encoded, one method per physical line.
        (d / "corpus_codes.txt").write_text(
            "\n".join(json.dumps(c) for c in PY_CORPUS_CODES) + "\n", encoding="utf-8"
        )
        return d

    def test_get_classes_filters_empty_class(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            out = fc.get_classes("P", "1", "pkg.mod", dataset="bugsinpy")
        lines = out.splitlines()
        assert "" not in lines  # no blank class for the module-level function
        assert "Outer" in lines
        assert "Outer.Inner" in lines

    def test_get_classes_function_only_module(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            out = fc.get_classes("P", "1", "pkg.only_funcs", dataset="bugsinpy")
        # A valid path with only module-level functions: informative, not a crash/recursion.
        assert "no classes" in out
        assert "get_methods_of_class" in out

    def test_get_methods_module_as_owner(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            # Passing the module path as the "class" lists its module-level functions.
            out = fc.get_methods("P", "1", "pkg.mod", dataset="bugsinpy")
        assert "top_level(a,b)" in out.splitlines()

    def test_find_method_module_level(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            out = fc.find_method("P", "1", "top_level", dataset="bugsinpy")
        assert any("top_level" in line for line in out.splitlines())

    def test_get_code_snippet_resolves(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            # corpus id round-trips via $ -> . (first occurrence only).
            out = fc.get_code_snippet("P", "1", "pkg.mod.Outer.method(self,x)", dataset="bugsinpy")
        # Faithful multi-line source with indentation preserved (not flattened).
        assert out == "def method(self, x):\n    return x"

    def test_get_code_snippet_multiline_block(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            out = fc.get_code_snippet("P", "1", "pkg.only_funcs.helper(z)", dataset="bugsinpy")
        assert out == "def helper(z):\n    if z:\n        return z\n    return None"
        ast.parse(out)  # round-trips to valid, parseable Python

    def test_get_paths(self, tmp_path: Path) -> None:
        d = self._corpus(tmp_path)
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=d):
            out = fc.get_paths("P", "1", dataset="bugsinpy")
        assert "pkg.mod" in out.splitlines()
        assert "pkg.only_funcs" in out.splitlines()


class TestValidBugGate:
    """cmd_corpus_bugsinpy skips bugs whose extraction did not validate."""

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            project="PySnooper", versions=[3], dataset="bugsinpy", force=False, keep_checkouts=True
        )

    def test_invalid_bug_skipped_no_corpus(self) -> None:
        import run_agent4sr

        bad = MagicMock(status="missing", error_msg="no method_signatures.csv", issues=[])
        repo = MagicMock()
        with (
            patch.object(run_agent4sr, "BugsInPyRepo", return_value=repo),
            patch("src.common.bip_gate.classify_bug", return_value=bad),
            patch.object(run_agent4sr, "save_corpus") as save_corpus,
        ):
            rc = run_agent4sr.cmd_corpus_bugsinpy(self._args(), MagicMock())

        assert rc == 0  # a gated skip is not an error
        save_corpus.assert_not_called()  # never an empty corpus for an unextracted bug
        repo.checkout.assert_not_called()  # and no wasted checkout

    def test_valid_bug_generates_corpus(self, tmp_path: Path) -> None:
        import run_agent4sr

        ok = MagicMock(status="ok")
        repo = MagicMock()
        with (
            patch.object(run_agent4sr, "BugsInPyRepo", return_value=repo),
            patch("src.common.bip_gate.classify_bug", return_value=ok),
            # No corpus on disk → the pre-checkout "exists" skip does not fire.
            patch("src.common.bip_gate.get_processed_dir", return_value=tmp_path),
            patch.object(run_agent4sr, "save_corpus") as save_corpus,
        ):
            rc = run_agent4sr.cmd_corpus_bugsinpy(self._args(), MagicMock())

        assert rc == 0
        repo.checkout.assert_called_once()
        repo.compile.assert_called_once()
        save_corpus.assert_called_once()
        # keep_checkouts=True → no cleanup.
        repo.remove_repo.assert_not_called()

    def test_existing_corpus_skipped_before_checkout(self, tmp_path: Path) -> None:
        """Resume path: a valid bug whose corpus already exists skips before checkout
        (when --force is off), so a long full-set run is cheaply resumable."""
        import run_agent4sr

        sr_dir = tmp_path / "FlexFL" / "SR"
        sr_dir.mkdir(parents=True)
        (sr_dir / "corpus_methods.txt").write_text("pkg.mod$foo(a)\n", encoding="utf-8")
        (sr_dir / "corpus_codes.txt").write_text("def foo(a): return a\n", encoding="utf-8")

        ok = MagicMock(status="ok")
        repo = MagicMock()
        with (
            patch.object(run_agent4sr, "BugsInPyRepo", return_value=repo),
            patch("src.common.bip_gate.classify_bug", return_value=ok),
            patch("src.common.bip_gate.get_processed_dir", return_value=tmp_path),
            patch.object(run_agent4sr, "save_corpus") as save_corpus,
        ):
            rc = run_agent4sr.cmd_corpus_bugsinpy(self._args(), MagicMock())

        assert rc == 0
        repo.checkout.assert_not_called()  # skipped before the expensive checkout
        save_corpus.assert_not_called()
