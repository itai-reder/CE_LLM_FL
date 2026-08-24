"""Integration tests verifying the exact output format of each pipeline step.

These tests use synthetic fixture data to exercise the full write path of each
FL module, then assert on the precise file format (headers, delimiters,
sort order, JSON keys, ID formats).

No Docker, Defects4J, or Ollama is required — all external dependencies are
mocked or replaced with in-memory data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent4sr.combine import _read_fl_csv, combine_candidates_for_bug
from src.agent4sr.corpus import save_corpus
from src.common.config import (
    BLUES_JSON,
    BLUES_STMT_SUSPS,
    BOOSTN_CSV,
    BOOSTN_JSON,
    SBFL_JSON,
    SBFL_STMT_SUSPS,
    SBIR_JSON,
    SBIR_STMT_SUSPS,
)

# =====================================================================
# Fixture data — a tiny synthetic "project"
# =====================================================================

SYNTHETIC_OCHIAI_CSV = """\
name;suspiciousness_value
org.example$Foo#doStuff(int):42;0.70710678118
org.example$Foo#doStuff(int):43;0.5
org.example$Foo#doStuff(int):44;0.0
org.example$Bar#init():10;0.35355339059
"""

SYNTHETIC_BUG_REPORT = {
    "url": "https://example.com/bugs/1",
    "title": "NPE in doStuff",
    "description": "Calling doStuff(null) throws NullPointerException",
    "raw": "NPE in doStuff\nCalling doStuff(null) throws NullPointerException",
}

SYNTHETIC_TRIGGER_TESTS = (
    "--- org.example.FooTest::testDoStuff\n"
    "java.lang.NullPointerException\n"
    "\tat org.example.Foo.doStuff(Foo.java:42)\n"
)

SYNTHETIC_FAULTS = "org.example.Foo 42\n"


@pytest.fixture
def synthetic_bug_dir(tmp_path: Path) -> Path:
    """Create a complete synthetic processed/<P>/<B>/ directory."""
    proc = tmp_path / "data" / "D4J" / "processed" / "TestProj" / "1"
    proc.mkdir(parents=True)

    # GZoltar SFL output
    sfl_dir = proc / "sfl" / "sfl" / "txt"
    sfl_dir.mkdir(parents=True)
    (sfl_dir / "ochiai.ranking.csv").write_text(SYNTHETIC_OCHIAI_CSV)

    # Bug report
    (proc / "bug_report.json").write_text(json.dumps(SYNTHETIC_BUG_REPORT))

    # Trigger tests
    (proc / "trigger_tests").write_text(SYNTHETIC_TRIGGER_TESTS)

    # Faults ground truth
    (proc / "faults.txt").write_text(SYNTHETIC_FAULTS)

    # dir.src.classes (for get_src_dir)
    (proc / "dir.src.classes").write_text("src/main/java")

    return proc


# =====================================================================
# Ochiai / SBFL output format
# =====================================================================


class TestOchiaiOutputFormat:
    """Verify SBFL produces the correct file formats."""

    def test_process_writes_correct_files(self, synthetic_bug_dir: Path) -> None:
        """SBFL should produce stmt-susps.txt and sbfl_ochiai.json."""
        from src.sbir.sbfl import SBFL

        ochiai_dir = synthetic_bug_dir / "Ochiai"
        ochiai_dir.mkdir(parents=True, exist_ok=True)

        sbfl = SBFL()
        with (
            patch(
                "src.sbir.sbfl.get_ochiai_ranking_dir",
                return_value=synthetic_bug_dir / "sfl" / "sfl" / "txt",
            ),
            patch("src.sbir.sbfl.get_ochiai_dir", return_value=ochiai_dir),
        ):
            sbfl.process_project("TestProj", "1")

        # --- Check stmt-susps.txt ---
        csv_path = ochiai_dir / SBFL_STMT_SUSPS
        assert csv_path.exists(), f"Expected {csv_path}"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["Statement", "Suspiciousness"]
            rows = list(reader)

        # Must be sorted descending by suspiciousness
        scores = [float(r["Suspiciousness"]) for r in rows]
        assert scores == sorted(scores, reverse=True), "Rows not sorted descending"

        # Statement IDs must use pkg.Class#lineNum format (NOT $ or :)
        for row in rows:
            stmt_id = row["Statement"]
            assert "#" in stmt_id, f"Statement ID missing '#': {stmt_id}"
            assert "$" not in stmt_id, f"Statement ID has '$': {stmt_id}"
            assert ":" not in stmt_id, f"Statement ID has ':': {stmt_id}"

        # First row should be the highest-scored statement
        assert rows[0]["Statement"] == "org.example.Foo#42"
        assert float(rows[0]["Suspiciousness"]) == pytest.approx(0.70710678118)

        # --- Check sbfl_ochiai.json ---
        json_path = ochiai_dir / SBFL_JSON
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "sbfl_scores" in data
        assert isinstance(data["sbfl_scores"], dict)
        # Keys should be statement IDs
        for key in data["sbfl_scores"]:
            assert "#" in key
            assert "$" not in key

    def test_ochiai_csv_readable_by_combine(self, synthetic_bug_dir: Path) -> None:
        """The Ochiai CSV should be parseable by combine._read_fl_csv."""
        from src.sbir.sbfl import SBFL

        ochiai_dir = synthetic_bug_dir / "Ochiai"
        ochiai_dir.mkdir(parents=True, exist_ok=True)

        sbfl = SBFL()
        with (
            patch(
                "src.sbir.sbfl.get_ochiai_ranking_dir",
                return_value=synthetic_bug_dir / "sfl" / "sfl" / "txt",
            ),
            patch("src.sbir.sbfl.get_ochiai_dir", return_value=ochiai_dir),
        ):
            sbfl.process_project("TestProj", "1")

        result = _read_fl_csv(ochiai_dir / SBFL_STMT_SUSPS, key_column="Statement")
        assert len(result) >= 1
        assert result[0] == "org.example.Foo#42"


# =====================================================================
# Blues output format
# =====================================================================


class TestBluesOutputFormat:
    """Verify Blues produces the correct file formats."""

    def test_blues_json_and_csv_format(self, synthetic_bug_dir: Path, tmp_path: Path) -> None:
        """Blues should produce stmt-susps-blues.txt and blues_scores.json."""
        from src.sbir.blues import Blues

        sbir_dir = synthetic_bug_dir / "SBIR"
        sbir_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal source directory with a Java file
        src_dir = tmp_path / "src" / "main" / "java" / "org" / "example"
        src_dir.mkdir(parents=True)
        (src_dir / "Foo.java").write_text(
            "package org.example;\n"
            "public class Foo {\n"
            "    public void doStuff(int x) {\n"
            "        if (x == 0) {\n"
            "            throw new RuntimeException();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )

        blues = Blues()
        with (
            patch("src.sbir.blues.get_processed_dir", return_value=synthetic_bug_dir),
            patch("src.sbir.blues.get_sbir_dir", return_value=sbir_dir),
            patch("src.sbir.blues.get_src_dir", return_value=tmp_path / "src" / "main" / "java"),
        ):
            blues.process_project("TestProj", "1")

        # --- CSV ---
        csv_path = sbir_dir / BLUES_STMT_SUSPS
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["Statement", "Suspiciousness"]
            rows = list(reader)

        assert len(rows) > 0
        for row in rows:
            assert "#" in row["Statement"]

        # --- JSON ---
        json_path = sbir_dir / BLUES_JSON
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "blues_scores" in data


# =====================================================================
# RAFL output format
# =====================================================================


class TestRAFLOutputFormat:
    """Verify RAFL produces the correct file formats."""

    def test_rafl_writes_sbir_susps(self, synthetic_bug_dir: Path) -> None:
        """RAFL should produce sbir-susps.txt and sbir_scores.json."""
        from src.sbir.rafl import RAFL
        from src.sbir.sbfl import SBFL

        # First produce Ochiai outputs
        ochiai_dir = synthetic_bug_dir / "Ochiai"
        ochiai_dir.mkdir(parents=True, exist_ok=True)
        sbfl = SBFL()
        with (
            patch(
                "src.sbir.sbfl.get_ochiai_ranking_dir",
                return_value=synthetic_bug_dir / "sfl" / "sfl" / "txt",
            ),
            patch("src.sbir.sbfl.get_ochiai_dir", return_value=ochiai_dir),
        ):
            sbfl.process_project("TestProj", "1")

        # Create a minimal Blues output (RAFL needs both rankings)
        sbir_dir = synthetic_bug_dir / "SBIR"
        sbir_dir.mkdir(parents=True, exist_ok=True)
        blues_csv = sbir_dir / BLUES_STMT_SUSPS
        blues_csv.write_text(
            "Statement,Suspiciousness\n"
            "org.example.Foo#42,1.0\n"
            "org.example.Foo#43,0.5\n"
            "org.example.Bar#10,0.2\n"
        )

        rafl = RAFL()
        with (
            patch("src.sbir.rafl.get_ochiai_dir", return_value=ochiai_dir),
            patch("src.sbir.rafl.get_sbir_dir", return_value=sbir_dir),
        ):
            rafl.process_project("TestProj", "1")

        # --- CSV ---
        csv_path = sbir_dir / SBIR_STMT_SUSPS
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["Statement", "Suspiciousness"]
            rows = list(reader)

        scores = [float(r["Suspiciousness"]) for r in rows]
        assert scores == sorted(scores, reverse=True), "RAFL output not sorted descending"

        # Top entry should be the one that ranks highest in both rankings
        assert rows[0]["Statement"] == "org.example.Foo#42"

        # --- JSON ---
        json_path = sbir_dir / SBIR_JSON
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "sbir_scores" in data

        # Scores should be Borda-normalised (max = 1.0)
        max_score = max(data["sbir_scores"].values())
        assert max_score == pytest.approx(1.0)


# =====================================================================
# BoostN output format
# =====================================================================


class TestBoostNOutputFormat:
    """Verify BoostN produces the correct file formats."""

    def test_boostn_writes_csv_and_json(self, synthetic_bug_dir: Path, tmp_path: Path) -> None:
        """BoostN should produce boostn-method-susps.csv and boostn.json."""
        from src.boostn.boostn import BoostN

        boostn_dir = synthetic_bug_dir / "BoostN"
        boostn_dir.mkdir(parents=True, exist_ok=True)

        # Create source directory with a Java file
        src_dir = tmp_path / "src" / "main" / "java" / "org" / "example"
        src_dir.mkdir(parents=True)
        (src_dir / "Foo.java").write_text(
            "package org.example;\n"
            "public class Foo {\n"
            "    public void doStuff(int x) {\n"
            "        if (x == 0) {\n"
            '            throw new RuntimeException("bad");\n'
            "        }\n"
            "        System.out.println(x);\n"
            "    }\n"
            "    public int getSize() {\n"
            "        return this.size;\n"
            "    }\n"
            "}\n"
        )

        boostn = BoostN()
        with (
            patch("src.common.config.get_processed_dir", return_value=synthetic_bug_dir),
            patch("src.common.config.get_boostn_dir", return_value=boostn_dir),
            patch(
                "src.common.config.get_src_dir",
                return_value=tmp_path / "src" / "main" / "java",
            ),
        ):
            boostn.process_project("TestProj", "1")

        # --- CSV ---
        csv_path = boostn_dir / BOOSTN_CSV
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["Signature", "Suspiciousness"]
            rows = list(reader)

        assert len(rows) >= 1
        scores = [float(r["Suspiciousness"]) for r in rows]
        assert scores == sorted(scores, reverse=True), "BoostN CSV not sorted descending"

        # Signatures use the corpus identity: pkg$Class.method(Params), no
        # trailing line numbers. Each must contain a '$' (package/class
        # separator) and end with a closing paren.
        for row in rows:
            sig = row["Signature"]
            assert "$" in sig, f"Expected '$' separator in corpus identity: {sig}"
            assert sig.endswith(")"), f"Expected corpus identity to end with ')': {sig}"

        # --- JSON ---
        json_path = boostn_dir / BOOSTN_JSON
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "boostn_scores" in data

        # CSV should be readable by combine._read_fl_csv with "Signature" key
        result = _read_fl_csv(csv_path, key_column="Signature")
        assert len(result) >= 1


# =====================================================================
# Agent4SR Corpus output format
# =====================================================================


class TestCorpusOutputFormat:
    """Verify corpus generation produces correct file formats."""

    def test_corpus_files_line_parallel(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """corpus_methods.txt and corpus_codes.txt must have same line count."""
        out_dir = tmp_path / "Agent4SR"
        out_dir.mkdir(parents=True)

        with (
            patch("src.agent4sr.corpus.get_src_dir", return_value=fixtures_dir),
            patch("src.agent4sr.corpus.get_sr_dir", return_value=out_dir),
        ):
            save_corpus("TestProj", "1")

        methods_lines = (out_dir / "corpus_methods.txt").read_text().strip().splitlines()
        codes_lines = (out_dir / "corpus_codes.txt").read_text().strip().splitlines()
        assert len(methods_lines) == len(codes_lines)
        assert len(methods_lines) > 0

    def test_corpus_id_uses_dollar_separator(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Corpus method IDs must use $ to separate package from class."""
        out_dir = tmp_path / "Agent4SR"
        out_dir.mkdir(parents=True)

        with (
            patch("src.agent4sr.corpus.get_src_dir", return_value=fixtures_dir),
            patch("src.agent4sr.corpus.get_sr_dir", return_value=out_dir),
        ):
            save_corpus("TestProj", "1")

        methods = (out_dir / "corpus_methods.txt").read_text().strip().splitlines()
        for mid in methods:
            assert "$" in mid, f"Corpus ID missing '$': {mid}"
            # Should NOT have .startLine.endLine suffix
            parts_after_paren = mid.split(")")
            if len(parts_after_paren) > 1:
                assert parts_after_paren[-1] == "", (
                    f"Corpus ID should not have line suffixes: {mid}"
                )


# =====================================================================
# Combine output format
# =====================================================================


class TestCombineOutputFormat:
    """Verify the combine step reads correct files and produces candidates.txt."""

    def test_combine_full_case(self, tmp_path: Path) -> None:
        """With all 4 FL methods available, candidates should have up to 20 entries."""
        # Set up FL output directories
        ochiai_dir = tmp_path / "Ochiai"
        ochiai_dir.mkdir()
        sbir_dir = tmp_path / "SBIR"
        sbir_dir.mkdir()
        boostn_dir = tmp_path / "BoostN"
        boostn_dir.mkdir()
        sr_dir = tmp_path / "SR"
        sr_dir.mkdir()
        model_dir = sr_dir / "test_model"
        model_dir.mkdir()

        # Ochiai (Statement column)
        (ochiai_dir / SBFL_STMT_SUSPS).write_text(
            "Statement,Suspiciousness\n"
            + "".join(f"org.example.Foo#{i},{1.0 - i * 0.1:.1f}\n" for i in range(10))
        )

        # SBIR (Statement column)
        (sbir_dir / SBIR_STMT_SUSPS).write_text(
            "Statement,Suspiciousness\n"
            + "".join(f"org.example.Bar#{i},{1.0 - i * 0.1:.1f}\n" for i in range(10))
        )

        # BoostN (Signature column — corpus identity)
        (boostn_dir / BOOSTN_CSV).write_text(
            "Signature,Suspiciousness\n"
            + "".join(f"org.example$Baz.method{i}(),{1.0 - i * 0.1:.1f}\n" for i in range(10))
        )

        # SR result
        sr_result = {
            "top5": [
                "org.example.Qux.alpha()",
                "org.example.Qux.beta()",
                "org.example.Qux.gamma()",
                "org.example.Qux.delta()",
                "org.example.Qux.epsilon()",
            ]
        }
        (model_dir / "sr_result.json").write_text(json.dumps(sr_result))

        # Corpus files for normalize_method_name
        (sr_dir / "corpus_methods.txt").write_text(
            "org.example$Qux.alpha()\norg.example$Qux.beta()\n"
            "org.example$Qux.gamma()\norg.example$Qux.delta()\n"
            "org.example$Qux.epsilon()\n"
        )
        (sr_dir / "corpus_codes.txt").write_text(
            "void alpha() {}\nvoid beta() {}\nvoid gamma() {}\nvoid delta() {}\nvoid epsilon() {}\n"
        )

        with (
            patch("src.agent4sr.combine.get_sbir_dir", return_value=sbir_dir),
            patch("src.agent4sr.combine.get_ochiai_dir", return_value=ochiai_dir),
            patch("src.agent4sr.combine.get_boostn_dir", return_value=boostn_dir),
            patch("src.agent4sr.combine.get_sr_model_dir", return_value=model_dir),
            patch("src.agent4sr.function_call.get_sr_dir", return_value=sr_dir),
        ):
            candidates = combine_candidates_for_bug("TestProj", "1", "test_model")

        # Should have: 5 SBIR + 5 Ochiai + 5 BoostN + 5 SR = 20
        assert len(candidates) == 20

        # First 5 should be SBIR (Statement format)
        for c in candidates[:5]:
            assert "#" in c, f"SBIR entry should have '#': {c}"

        # Next 5 should be Ochiai (Statement format)
        for c in candidates[5:10]:
            assert "#" in c, f"Ochiai entry should have '#': {c}"

        # Next 5 should be BoostN (Signature format with line numbers)
        for c in candidates[10:15]:
            assert "method" in c, f"BoostN entry should have method name: {c}"

        # Last 5 should be SR (corpus-normalised names)
        for c in candidates[15:20]:
            assert "Qux" in c, f"SR entry should have Qux: {c}"

    def test_combine_fallback_no_boostn(self, tmp_path: Path) -> None:
        """Without BoostN, should fall back to Ochiai[:15]."""
        ochiai_dir = tmp_path / "Ochiai"
        ochiai_dir.mkdir()
        sr_dir = tmp_path / "SR"
        sr_dir.mkdir()
        model_dir = sr_dir / "test_model"
        model_dir.mkdir()

        (ochiai_dir / SBFL_STMT_SUSPS).write_text(
            "Statement,Suspiciousness\n"
            + "".join(f"org.example.Foo#{i},{1.0 - i * 0.01:.2f}\n" for i in range(20))
        )

        (model_dir / "sr_result.json").write_text(json.dumps({"top5": ["a", "b"]}))
        (sr_dir / "corpus_methods.txt").write_text("org.example$Foo.a()\n")
        (sr_dir / "corpus_codes.txt").write_text("void a() {}\n")

        with (
            patch("src.agent4sr.combine.get_sbir_dir", side_effect=FileNotFoundError),
            patch("src.agent4sr.combine.get_ochiai_dir", return_value=ochiai_dir),
            patch("src.agent4sr.combine.get_boostn_dir", side_effect=FileNotFoundError),
            patch("src.agent4sr.combine.get_sr_model_dir", return_value=model_dir),
            patch("src.agent4sr.function_call.get_sr_dir", return_value=sr_dir),
        ):
            candidates = combine_candidates_for_bug("TestProj", "1", "test_model")

        # Fallback: Ochiai[:15] + SR[:5] = up to 17
        assert len(candidates) >= 15


# =====================================================================
# Cross-format: faults.txt vs FL ranking ID translation
# =====================================================================


class TestFaultIdTranslation:
    """Verify the ID format difference between faults.txt and FL rankings."""

    def test_fault_format_differs_from_statement_id(self) -> None:
        """faults.txt uses 'pkg.Class lineNum', rankings use 'pkg.Class#lineNum'."""
        fault_line = "org.example.Foo 42"
        parts = fault_line.split()
        assert len(parts) == 2
        class_name, line_num = parts

        # Convert to statement ID format
        stmt_id = f"{class_name}#{line_num}"
        assert stmt_id == "org.example.Foo#42"

        # Verify the formats are different
        assert " " in fault_line
        assert "#" not in fault_line
        assert "#" in stmt_id
        assert " " not in stmt_id

    def test_gzoltar_id_conversion(self) -> None:
        """GZoltar pkg$Class#method():line -> CEFL pkg.Class#line."""
        from src.sbir.sbfl import SBFL

        # The _to_statement_id method converts GZoltar format
        gzoltar_id = "org.example$Foo#doStuff(int):42"
        # Expected: org.example.Foo#42
        stmt_id = SBFL._to_statement_id(gzoltar_id)
        assert stmt_id == "org.example.Foo#42"
        assert "$" not in stmt_id
        assert ":" not in stmt_id
        assert "doStuff" not in stmt_id  # method name stripped
