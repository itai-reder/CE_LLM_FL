"""Tests for dataset-aware dispatch wrappers in src.common.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common import config

# Resolvers still scoped to Defects4J. ``get_repo_dir`` stays D4J-only: BugsInPy
# resolves its checkout via BugsInPyRepo, not data/<bench>/repos.
_HELPERS_NO_MODEL = [
    "get_repo_dir",
]

# Tier-2 FL output dirs resolve for BugsInPy too: same subdir names under
# the BIP processed root, including the FlexFL/SR|LR family.
_HELPERS_BIP_SUPPORTED = [
    ("get_processed_dir", None),
    ("get_ochiai_dir", config.OCHIAI_SUBDIR),
    ("get_boostn_dir", config.BOOSTN_SUBDIR),
    ("get_sbir_dir", config.SBIR_SUBDIR),
    ("get_sr_dir", config.FLEXFL_SR_SUBDIR),
    ("get_rankings_dir", config.FLEXFL_SR_SUBDIR / "rankings"),
    ("get_lr_dir", config.FLEXFL_LR_SUBDIR),
    ("get_lr_checkpoints_dir", config.FLEXFL_LR_SUBDIR / "checkpoints"),
]


@pytest.mark.parametrize("name", _HELPERS_NO_MODEL)
def test_wrapper_rejects_non_d4j(name: str) -> None:
    fn = getattr(config, name)
    with pytest.raises(NotImplementedError, match="not supported yet"):
        fn("Lang", 1, dataset="bugsinpy")


@pytest.mark.parametrize(("name", "subdir"), _HELPERS_BIP_SUPPORTED)
def test_fl_output_dir_resolves_bugsinpy(name: str, subdir) -> None:
    """The tier-2 FL output dirs resolve under the BIP processed root."""
    fn = getattr(config, name)
    bip_base = config.LAYOUT.processed_root("bugsinpy") / "youtube-dl" / "2"
    expected = bip_base if subdir is None else bip_base / subdir
    assert fn("youtube-dl", 2, dataset="bugsinpy") == expected


def test_fl_output_dir_rejects_unknown_benchmark() -> None:
    with pytest.raises(NotImplementedError, match="not supported yet"):
        config.get_ochiai_dir("Lang", 1, dataset="nosuchbenchmark")


def test_sr_model_dir_resolves_bugsinpy() -> None:
    """The SR model dir resolves under the BIP processed root."""
    expected = (
        config.LAYOUT.processed_root("bugsinpy")
        / "youtube-dl"
        / "2"
        / config.FLEXFL_SR_SUBDIR
        / "Agent4SR"
        / config._model_slug("llama3.1:8b")
    )
    assert config.get_sr_model_dir("youtube-dl", 2, "llama3.1:8b", dataset="bugsinpy") == expected


@pytest.mark.parametrize(
    "name",
    [
        "get_processed_dir",
        "get_sr_dir",
        "get_rankings_dir",
    ],
)
def test_wrapper_delegates_to_d4j(name: str) -> None:
    """The dispatch wrapper must return the same Path as the _d4j body for D4J."""
    wrapper = getattr(config, name)
    impl = getattr(config, f"{name}_d4j")
    assert wrapper("Lang", 1) == impl("Lang", 1)
    assert wrapper("Lang", 1, dataset="defects4j") == impl("Lang", 1)


@pytest.mark.parametrize(
    ("name", "subdir"),
    [
        ("get_ochiai_dir", config.OCHIAI_SUBDIR),
        ("get_boostn_dir", config.BOOSTN_SUBDIR),
        ("get_sbir_dir", config.SBIR_SUBDIR),
    ],
)
def test_fl_output_dir_d4j_unchanged(name: str, subdir) -> None:
    """The D4J FL output dirs keep resolving to data/D4J/processed/.../FL/<subdir>."""
    fn = getattr(config, name)
    d4j_base = config.LAYOUT.processed_root("defects4j") / "Lang" / "1"
    assert fn("Lang", 1) == d4j_base / subdir
    assert fn("Lang", 1, dataset="defects4j") == d4j_base / subdir


def test_sr_model_dir_delegates_to_d4j() -> None:
    assert config.get_sr_model_dir("Lang", 1, "llama3.1:8b") == config.get_sr_model_dir_d4j(
        "Lang", 1, "llama3.1:8b"
    )


def test_sr_model_dir_uses_model_id_verbatim() -> None:
    """When ``model_id`` is given the on-disk dir name is used verbatim."""
    p = config.get_sr_model_dir("Lang", 1, "llama3.1:8b", model_id="llama3_1_8b__1")
    assert p.name == "llama3_1_8b__1"


def test_sr_model_dir_falls_back_to_slug() -> None:
    p = config.get_sr_model_dir("Lang", 1, "llama3.1:8b")
    assert p.name == config._model_slug("llama3.1:8b")


def test_processed_dir_under_layout_root() -> None:
    """Default branch routes to data/D4J/processed/<Project>/<BugId>/."""
    p = config.get_processed_dir("Lang", 1)
    assert p == config.LAYOUT.processed_root("defects4j") / "Lang" / "1"


# ---------------------------------------------------------------------------
# BugsInPy support: get_src_dir + coverage-layout resolvers
# ---------------------------------------------------------------------------


def test_src_dir_bugsinpy_delegates_to_repo(monkeypatch) -> None:
    """get_src_dir(dataset=bugsinpy) delegates to BugsInPyRepo.get_src_class_dir."""
    from pathlib import Path

    import src.extraction.bugsinpy as bip

    monkeypatch.setattr(
        bip.BugsInPyRepo, "get_src_class_dir", lambda self: Path("/fake/import/root")
    )
    assert config.get_src_dir("youtube-dl", 2, dataset="bugsinpy") == Path("/fake/import/root")


@pytest.mark.parametrize(
    ("resolver", "d4j_subdir", "bip_subdir"),
    [
        ("get_coverage_dir", config.SFL_SUBDIR, config.FAUXPY_COVERAGE_SUBDIR),
        ("get_ochiai_ranking_dir", config.SFL_SUBDIR, config.FAUXPY_REPORTS_SUBDIR),
    ],
)
def test_coverage_resolvers(resolver: str, d4j_subdir, bip_subdir) -> None:
    fn = getattr(config, resolver)
    d4j_base = config.LAYOUT.processed_root("defects4j") / "Lang" / "1"
    bip_base = config.LAYOUT.processed_root("bugsinpy") / "youtube-dl" / "2"
    assert fn("Lang", 1) == d4j_base / d4j_subdir
    assert fn("youtube-dl", 2, dataset="bugsinpy") == bip_base / bip_subdir


def test_fauxpy_raw_dir() -> None:
    base = config.LAYOUT.processed_root("bugsinpy") / "youtube-dl" / "2"
    assert config.get_fauxpy_raw_dir("youtube-dl", 2) == base / config.FAUXPY_RAW_SUBDIR


# ---------------------------------------------------------------------------
# Traditional FL: SBFL/Blues/BoostN/RAFL accept ``dataset=bugsinpy``.
# They no longer reject at dispatch — they read BIP inputs and fail only on
# missing data (FileNotFoundError), not NotImplementedError. End-to-end behavior
# is covered in test_bugsinpy_fl.py and the on-disk verification.
# ---------------------------------------------------------------------------


def test_sbfl_process_project_accepts_bugsinpy(tmp_path, monkeypatch) -> None:
    """SBFL no longer rejects bugsinpy; it resolves the BIP ranking path and fails
    only on the missing ranking file (FileNotFoundError, not NotImplementedError)."""
    import src.sbir.sbfl as sbfl_mod
    from src.sbir.sbfl import SBFL

    # Redirect the output + ranking dirs to tmp so the test does not touch data/.
    monkeypatch.setattr(sbfl_mod, "get_ochiai_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(sbfl_mod, "get_ochiai_ranking_dir", lambda *a, **k: tmp_path)

    with pytest.raises(FileNotFoundError):
        SBFL().process_project("Lang", 1, dataset="bugsinpy")


# ---------------------------------------------------------------------------
# Agent4SR / rankings / validation propagate dataset kwarg
# ---------------------------------------------------------------------------


def test_save_corpus_threads_bugsinpy(tmp_path, monkeypatch) -> None:
    """save_corpus accepts bugsinpy; it threads the dataset to
    generate_corpus and writes both corpus files to the resolved SR dir."""
    import src.agent4sr.corpus as corpus_mod
    from src.agent4sr.corpus import Corpus, save_corpus

    out_dir = tmp_path / "SR"
    out_dir.mkdir()
    monkeypatch.setattr(corpus_mod, "get_sr_dir", lambda *a, **k: out_dir)
    monkeypatch.setattr(
        corpus_mod,
        "generate_corpus",
        lambda project, bug_id, *, dataset: Corpus(
            method_ids=["pkg.mod$foo(a)"], raw_codes=["def foo(a): return a"]
        ),
    )
    save_corpus("PySnooper", 3, dataset="bugsinpy", skip_existing=False)
    assert (out_dir / "corpus_methods.txt").read_text(encoding="utf-8") == "pkg.mod$foo(a)\n"


def test_load_bug_inputs_bugsinpy_io_is_benchmark_agnostic() -> None:
    """load_bug_inputs reads tier-2/3 artifacts by path; with ``get_processed_dir``
    resolving for bugsinpy, it does not reject — it degrades gracefully
    when the per-bug artifacts are absent."""
    from src.agent4sr.io import load_bug_inputs

    inputs = load_bug_inputs("Lang", 1, dataset="bugsinpy")
    assert inputs.trigger_test == ""


def test_corpus_methods_path_resolves_bugsinpy() -> None:
    """corpus_methods.txt path resolves under the BIP SR dir."""
    from src.agent4sr.function_call import corpus_methods_path

    expected = (
        config.LAYOUT.processed_root("bugsinpy")
        / "youtube-dl"
        / "2"
        / config.FLEXFL_SR_SUBDIR
        / "corpus_methods.txt"
    )
    assert corpus_methods_path("youtube-dl", 2, dataset="bugsinpy") == expected


# The combine/rankings writers are benchmark-agnostic readers: with the FlexFL/SR
# resolvers generalized, they do not reject bugsinpy at the config
# layer — they degrade gracefully when the per-bug SR/FL artifacts are absent (the
# Phase-2 ``run``/``combine`` CLI still gates them via _require_d4j_dataset). The
# tests redirect output dirs to tmp so they touch no real data.


def test_write_candidates_benchmark_agnostic(tmp_path, monkeypatch) -> None:
    import src.agent4sr.combine as combine_mod
    from src.agent4sr.combine import write_candidates

    monkeypatch.setattr(combine_mod, "get_sr_model_dir", lambda *a, **k: tmp_path)
    # No SR/FL inputs present → degrades to an empty candidate file, not a raise.
    out = write_candidates("youtube-dl", 2, dataset="bugsinpy")
    assert out.parent == tmp_path or tmp_path in out.parents


def test_generate_all_rankings_benchmark_agnostic(tmp_path, monkeypatch) -> None:
    import src.common.rankings as rankings_mod
    from src.common.rankings import generate_all_rankings

    monkeypatch.setattr(rankings_mod, "get_processed_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(rankings_mod, "get_rankings_dir", lambda *a, **k: tmp_path)
    # method_signatures.csv absent → warn-and-skip, returns an (empty) dict, no raise.
    assert generate_all_rankings("youtube-dl", 2, dataset="bugsinpy") == {}


def test_generate_top15_benchmark_agnostic(tmp_path, monkeypatch) -> None:
    import src.common.rankings as rankings_mod
    from src.common.rankings import generate_top15

    monkeypatch.setattr(rankings_mod, "get_processed_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(rankings_mod, "get_rankings_dir", lambda *a, **k: tmp_path)
    assert isinstance(generate_top15("youtube-dl", 2, dataset="bugsinpy"), Path)


def test_generate_top20_benchmark_agnostic(tmp_path, monkeypatch) -> None:
    import src.common.rankings as rankings_mod
    from src.common.rankings import generate_top20

    monkeypatch.setattr(rankings_mod, "get_processed_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(rankings_mod, "get_rankings_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(rankings_mod, "get_sr_model_dir", lambda *a, **k: tmp_path)
    assert isinstance(generate_top20("youtube-dl", 2, "llama3.1:8b", dataset="bugsinpy"), Path)


def test_validate_extraction_outputs_bugsinpy_supported(tmp_path) -> None:
    """BugsInPy is supported: an empty dir yields missing-file issues, not a raise."""
    from src.extraction.validation import validate_extraction_outputs

    issues = validate_extraction_outputs(
        tmp_path,
        expect_gzoltar=False,
        expect_faults=False,
        expect_bug_report=False,
        dataset="bugsinpy",
    )
    assert any(i["file"] == "method_signatures.csv" for i in issues)


def test_validate_extraction_outputs_rejects_unknown(tmp_path) -> None:
    from src.extraction.validation import validate_extraction_outputs

    with pytest.raises(NotImplementedError):
        validate_extraction_outputs(
            tmp_path,
            expect_gzoltar=False,
            expect_faults=False,
            expect_bug_report=False,
            dataset="nosuchbenchmark",
        )
