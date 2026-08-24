"""
Shared configuration for FL methods.

All paths are relative to the CEFL project root.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.core.layout import DatasetLayout, normalize_benchmark_name

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# fl_methods/src/common/ → fl_methods/src/ → fl_methods/ → CEFL/
_THIS_DIR = Path(__file__).resolve().parent  # fl_methods/src/common/
FL_METHODS_DIR = _THIS_DIR.parent.parent  # fl_methods/
PROJECT_ROOT = FL_METHODS_DIR.parent  # CEFL/

DEFAULT_BENCHMARK = "defects4j"

# CEFL_D4J_WORKSPACE relocates the D4J workspace (the dir bound to /workspace
# inside the container, and the parent of processed/repos/fixed/deps for D4J),
# e.g. to place the heavyweight benchmark data on a faster or larger volume.
_D4J_DEFAULT_ROOT = PROJECT_ROOT / "data" / "D4J"
_d4j_workspace_env = os.environ.get("CEFL_D4J_WORKSPACE")
_D4J_WORKSPACE = Path(_d4j_workspace_env).resolve() if _d4j_workspace_env else _D4J_DEFAULT_ROOT

# CEFL_BIP_WORKSPACE relocates the BugsInPy workspace (the dir bound to /workspace
# inside the BugsInPy container, and the parent of processed/repos/fixed/deps for
# BugsInPy). It mirrors CEFL_D4J_WORKSPACE. The container wrapper
# (utils/docker/bugsinpy/cmd-docker-bip.sh) reads the SAME env var to decide what to
# mount at /workspace, so this override and the wrapper must stay in lockstep —
# otherwise repos_root("bugsinpy") and the container mount silently diverge.
_BIP_DEFAULT_ROOT = PROJECT_ROOT / "data" / "BIP"
_bip_workspace_env = os.environ.get("CEFL_BIP_WORKSPACE")
_BIP_WORKSPACE = Path(_bip_workspace_env).resolve() if _bip_workspace_env else _BIP_DEFAULT_ROOT

_benchmark_overrides: dict[str, Path] = {}
if _d4j_workspace_env:
    _benchmark_overrides["D4J"] = _D4J_WORKSPACE
if _bip_workspace_env:
    _benchmark_overrides["BIP"] = _BIP_WORKSPACE

LAYOUT = DatasetLayout(PROJECT_ROOT, benchmark_overrides=_benchmark_overrides)


def get_d4j_workspace_root() -> Path:
    """Return the active host-side D4J workspace (CEFL_D4J_WORKSPACE or default)."""
    return _D4J_WORKSPACE


def d4j_workspace_is_relocated() -> bool:
    """True when CEFL_D4J_WORKSPACE points outside ``PROJECT_ROOT/data/D4J``."""
    return _D4J_WORKSPACE.resolve() != _D4J_DEFAULT_ROOT.resolve()


def get_bip_workspace_root() -> Path:
    """Return the active host-side BugsInPy workspace (CEFL_BIP_WORKSPACE or default)."""
    return _BIP_WORKSPACE


def bip_workspace_is_relocated() -> bool:
    """True when CEFL_BIP_WORKSPACE points outside ``PROJECT_ROOT/data/BIP``."""
    return _BIP_WORKSPACE.resolve() != _BIP_DEFAULT_ROOT.resolve()


# GZoltar JARs (utils/java/)
GZOLTAR_DIR = PROJECT_ROOT / "utils" / "java"
GZOLTAR_CLI_JAR_NAME = (
    "gzoltar-cli.jar"  # Originally "com.gzoltar.cli-1.7.4-SNAPSHOT-jar-with-dependencies.jar"
)
GZOLTAR_AGENT_JAR_NAME = (
    "gzoltar-agent.jar"  # Originally "com.gzoltar.agent.rt-1.7.4-SNAPSHOT-all.jar"
)
GZOLTAR_CLI_JAR = GZOLTAR_DIR / GZOLTAR_CLI_JAR_NAME
GZOLTAR_AGENT_JAR = GZOLTAR_DIR / GZOLTAR_AGENT_JAR_NAME

# Stopwords files (shared between BoostN and Blues). The Java list is the default;
# BugsInPy (Python) selects the Python list via get_stopwords_file(dataset).
STOPWORDS_FILE = FL_METHODS_DIR / "src" / "boostn" / "StopwordsPlusJava.txt"
STOPWORDS_FILE_PYTHON = FL_METHODS_DIR / "src" / "boostn" / "StopwordsPlusPython.txt"


# ---------------------------------------------------------------------------
# Docker / Defects4J
# ---------------------------------------------------------------------------

CONTAINER_NAME = os.environ.get("CEFL_D4J_CONTAINER", "defects4j-cefl-container")
CONTAINER_USER = "1000:1000"
CONTAINER_WORKSPACE = "/workspace"  # mount point inside the container

# Defects4J installation inside the container
D4J_CONTAINER_DIR = "/defects4j"
D4J_JUNIT_CONTAINER = "/defects4j/framework/projects/lib/junit-4.12-hamcrest-1.3.jar"

# ---------------------------------------------------------------------------
# Docker / BugsInPy
# ---------------------------------------------------------------------------

# Generic conda-init-aware passthrough wrapper. BugsInPyRepo shells to this the same
# way D4J ops shell to the d4j container (see src.extraction.bugsinpy.run_bip).
BIP_WRAPPER = PROJECT_ROOT / "utils" / "docker" / "bugsinpy" / "cmd-docker-bip.sh"
BIP_CONTAINER_WORKSPACE = "/workspace"  # host BugsInPy workspace mount point
# BugsInPy project/bug index. The host copy is gitignored (cloned at Stage-0 setup),
# so callers fall back to BIP_INDEX_CONTAINER via the wrapper when it is absent.
BIP_INDEX_CSV = BIP_WRAPPER.parent / "BugsInPy" / "projects" / "bugsinpy-index.csv"
BIP_INDEX_CONTAINER = "/BugsInPy/projects/bugsinpy-index.csv"

# ---------------------------------------------------------------------------
# GZoltar SFL output paths (relative to processed dir)
# ---------------------------------------------------------------------------

SFL_SUBDIR = Path("sfl") / "sfl" / "txt"
OCHIAI_RANKING_FILE = "ochiai.ranking.csv"
SPECTRA_FILE = "spectra.csv"
MATRIX_FILE = "matrix.txt"
TESTS_FILE = "tests.csv"

# ---------------------------------------------------------------------------
# BugsInPy / FauxPy coverage layout (relative to processed dir)
# ---------------------------------------------------------------------------
#
# BugsInPy uses a FauxPy-provenance tree rather than cloning GZoltar's
# ``sfl/sfl/txt/``. The canonical schema (spectra/matrix/tests/ochiai) keeps the
# SAME file basenames + column formats as D4J — only the *location* differs and is
# resolved through ``get_coverage_dir`` / ``get_ochiai_ranking_dir`` per benchmark.
#   FauxPy/raw/      raw FauxPy report (fauxpy.db, Scores_*.csv) — intermediate only
#   FauxPy/coverage/ canonical spectra.csv / matrix.txt / tests.csv
#   FauxPy/reports/  canonical ochiai.ranking.csv
FAUXPY_RAW_SUBDIR = Path("FauxPy") / "raw"
FAUXPY_COVERAGE_SUBDIR = Path("FauxPy") / "coverage"
FAUXPY_REPORTS_SUBDIR = Path("FauxPy") / "reports"

# ---------------------------------------------------------------------------
# Output subdirectory names
# ---------------------------------------------------------------------------

OCHIAI_SUBDIR = Path("FL") / "Ochiai"  # processed_dir/FL/Ochiai/
BOOSTN_SUBDIR = Path("FL") / "BoostN"  # processed_dir/FL/BoostN/
SBIR_SUBDIR = Path("FL") / "SBIR"  # processed_dir/FL/SBIR/
FLEXFL_SR_SUBDIR = Path("FlexFL") / "SR"  # processed_dir/FlexFL/SR/
FLEXFL_LR_SUBDIR = Path("FlexFL") / "LR"  # processed_dir/FlexFL/LR/

# ---------------------------------------------------------------------------
# Output file names produced by our scripts
# ---------------------------------------------------------------------------

# Ochiai (SBFL) outputs — written to FL/Ochiai/
SBFL_STMT_SUSPS = "stmt-susps.txt"  # SBFL Ochiai stmt ranking
SBFL_JSON = "sbfl_ochiai.json"  # SBFL scores as JSON
# Blues intermediate outputs — written to FL/SBIR/
BLUES_STMT_SUSPS = "stmt-susps-blues.txt"  # Blues stmt ranking
BLUES_JSON = "blues_scores.json"  # Blues scores as JSON
SBIR_STMT_SUSPS = "sbir-susps.txt"  # Final SBIR aggregated ranking
SBIR_JSON = "sbir_scores.json"  # Final SBIR aggregated scores as JSON
BOOSTN_JSON = "boostn.json"  # BoostN scores as JSON
BOOSTN_CSV = "boostn-method-susps.csv"  # BoostN method ranking CSV


# ---------------------------------------------------------------------------
# Repo-level result / evaluation / transcript / log roots
# ---------------------------------------------------------------------------
#
# These four trees are anchored on PROJECT_ROOT and are never relocated by the
# workspace env vars. ``results/`` and ``evaluation/`` are committed to git and
# hold everything needed to (re)run the evaluation without any processed data;
# ``transcripts/`` and ``logs/`` are gitignored.


def get_results_root(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return ``results/<Benchmark>/`` — slim per-bug evaluation inputs."""
    canonical = _require_supported(dataset)
    return PROJECT_ROOT / "results" / canonical


def get_results_bug_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return ``results/<Benchmark>/<Project>/<BugId>/`` (not created)."""
    return get_results_root(dataset) / project / str(bug_id)


def get_results_meta_dir(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return ``results/<Benchmark>/_meta/`` for benchmark-level snapshots.

    Holds files shared across bugs, e.g. the BugsInPy extraction ``audit.csv``
    and a ``bugsinpy-index.csv`` snapshot so project/bug enumeration works
    without the benchmark clone or a running container.
    """
    return get_results_root(dataset) / "_meta"


def get_evaluation_root(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return ``evaluation/<Benchmark>/`` — cross-bug evaluation CSVs."""
    canonical = _require_supported(dataset)
    return PROJECT_ROOT / "evaluation" / canonical


def get_transcripts_bug_dir(
    project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK
) -> Path:
    """Return ``transcripts/<Benchmark>/<Project>/<BugId>/`` (not created).

    Full LLM interaction records (``lr_result.json`` / ``sr_result.json``)
    extracted from the processed tree; gitignored.
    """
    canonical = _require_supported(dataset)
    return PROJECT_ROOT / "transcripts" / canonical / project / str(bug_id)


def get_logs_dir(stage: str, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return ``logs/<Benchmark>/<stage>/``, creating it if needed."""
    canonical = _require_supported(dataset)
    d = PROJECT_ROOT / "logs" / canonical / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# Two layers exist here:
#   * ``*_d4j`` functions hold the Defects4J-specific implementation. They
#     resolve paths via ``LAYOUT`` for the canonical "defects4j" benchmark and
#     do not accept a ``dataset`` argument.
#   * The bare-name wrappers (``get_processed_dir``, …) accept
#     ``dataset="defects4j"`` and dispatch to the ``*_d4j`` implementation, or
#     raise ``NotImplementedError`` for any other benchmark. New adapters plug
#     in by adding sibling implementations and extending the dispatchers.


def _require_d4j(dataset: str) -> None:
    if dataset != DEFAULT_BENCHMARK:
        raise NotImplementedError(
            f"Dataset {dataset!r} is not supported yet; only {DEFAULT_BENCHMARK!r}."
        )


def get_benchmark_data_dir(benchmark: str = DEFAULT_BENCHMARK) -> Path:
    """Return benchmark data root under ``data/<Benchmark>/``."""
    return LAYOUT.benchmark_root(benchmark)


def get_benchmark_processed_root(benchmark: str = DEFAULT_BENCHMARK) -> Path:
    """Return benchmark processed root under ``data/<Benchmark>/processed``."""
    return LAYOUT.processed_root(benchmark)


def get_benchmark_repos_root(benchmark: str = DEFAULT_BENCHMARK) -> Path:
    """Return benchmark repos root under ``data/<Benchmark>/repos``."""
    return LAYOUT.repos_root(benchmark)


def get_benchmark_fixed_root(benchmark: str = DEFAULT_BENCHMARK) -> Path:
    """Return benchmark fixed root under ``data/<Benchmark>/fixed``."""
    return LAYOUT.fixed_root(benchmark)


# Default character substitutions applied by ``_model_slug``. Extend this map
# to handle additional characters that are unsafe in filesystem paths (or pass
# a custom map via the ``slug_map`` argument).
_SLUG_MAP: dict[str, str] = {":": "_"}


def _model_slug(model: str, slug_map: dict[str, str] = _SLUG_MAP) -> str:
    """Convert a model name to a filesystem-safe slug.

    Each ``key`` in ``slug_map`` is replaced by its ``value`` in order. The
    default map turns ``llama3.1:8b`` into ``llama3.1_8b``. Pass a custom map
    to override or extend the substitutions.
    """
    out = model
    for src, dst in slug_map.items():
        out = out.replace(src, dst)
    return out


# ---- Defects4J-specific implementations (Type A) --------------------------


def get_processed_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return data/D4J/processed/<Project>/<BugId>/."""
    return LAYOUT.processed_root(DEFAULT_BENCHMARK) / project / str(bug_id)


def get_sr_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return processed_dir/FlexFL/SR/ for Agent4SR corpus and model outputs."""
    d = get_processed_dir_d4j(project, bug_id) / FLEXFL_SR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sr_model_dir_d4j(
    project: str, bug_id: str | int, model: str, *, model_id: str | None = None
) -> Path:
    """Return processed_dir/FlexFL/SR/Agent4SR/<dir_name>/ for a specific model's outputs.

    ``dir_name`` is ``model_id`` when provided (allows the tracker to assign a
    suffixed slug like ``llama3_1_8b__1`` for a second config), otherwise falls
    back to ``_model_slug(model)``.
    """
    name = model_id if model_id is not None else _model_slug(model)
    d = get_sr_dir_d4j(project, bug_id) / "Agent4SR" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rankings_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return processed_dir/FlexFL/SR/rankings/ for standardised ranking outputs."""
    d = get_sr_dir_d4j(project, bug_id) / "rankings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lr_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return processed_dir/FlexFL/LR/ for Agent4LR per-model outputs."""
    d = get_processed_dir_d4j(project, bug_id) / FLEXFL_LR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_repo_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return the checked-out buggy D4J repo directory.

    Tries <Project>/<BugId>_buggy first, then <Project>/<BugId>.
    """
    repos_root = LAYOUT.repos_root(DEFAULT_BENCHMARK)
    buggy = repos_root / project / f"{bug_id}_buggy"
    if buggy.exists():
        return buggy
    plain = repos_root / project / str(bug_id)
    if plain.exists():
        return plain
    raise FileNotFoundError(f"No repo found for {project}-{bug_id} in {repos_root / project}")


def get_src_dir_d4j(project: str, bug_id: str | int) -> Path:
    """Return the source classes directory for a given D4J bug.

    Reads from the Defects4J export file dir.src.classes if available,
    otherwise falls back to common defaults.
    """
    processed = get_processed_dir_d4j(project, bug_id)
    src_file = processed / "dir.src.classes"
    if src_file.exists():
        rel = src_file.read_text().strip()
        return get_repo_dir_d4j(project, bug_id) / rel

    # Fallback heuristics
    repo = get_repo_dir_d4j(project, bug_id)
    for candidate in ["src/main/java", "src/java", "source", "src"]:
        if (repo / candidate).is_dir():
            return repo / candidate
    return repo


# ---- BugsInPy-specific implementations (Type A) ---------------------------


def get_src_dir_bugsinpy(project: str, bug_id: str | int) -> Path:
    """Return the source-classes import root for a BugsInPy bug (buggy version).

    Delegates to :meth:`BugsInPyRepo.get_src_class_dir` (the single source of truth
    for import-root resolution: ``pythonpath`` if it exists, else the layout-derived
    root). Requires the checkout on disk.
    """
    from src.extraction.bugsinpy import BugsInPyRepo

    return BugsInPyRepo(project, int(bug_id)).get_src_class_dir()


# ---- Coverage layout resolvers (per-benchmark, tier-1) --------------------


def coverage_subdir(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return the spectra/matrix/tests subdir (relative to processed dir)."""
    canonical = normalize_benchmark_name(dataset)
    if canonical == "D4J":
        return SFL_SUBDIR
    if canonical == "BIP":
        return FAUXPY_COVERAGE_SUBDIR
    raise NotImplementedError(f"Dataset {dataset!r} is not supported yet.")


def ranking_subdir(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return the ochiai.ranking.csv subdir (relative to processed dir)."""
    canonical = normalize_benchmark_name(dataset)
    if canonical == "D4J":
        return SFL_SUBDIR
    if canonical == "BIP":
        return FAUXPY_REPORTS_SUBDIR
    raise NotImplementedError(f"Dataset {dataset!r} is not supported yet.")


def get_coverage_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Dir holding spectra/matrix/tests (GZoltar ``sfl/sfl/txt`` for D4J;
    ``FauxPy/coverage`` for BugsInPy)."""
    base = LAYOUT.processed_root(dataset) / project / str(bug_id)
    return base / coverage_subdir(dataset)


def get_ochiai_ranking_dir(
    project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK
) -> Path:
    """Dir holding ``ochiai.ranking.csv`` (``sfl/sfl/txt`` for D4J;
    ``FauxPy/reports`` for BugsInPy)."""
    base = LAYOUT.processed_root(dataset) / project / str(bug_id)
    return base / ranking_subdir(dataset)


def get_fauxpy_raw_dir(project: str, bug_id: str | int) -> Path:
    """Dir holding the untouched FauxPy report (BugsInPy only; intermediate)."""
    base = LAYOUT.processed_root("bugsinpy") / project / str(bug_id)
    return base / FAUXPY_RAW_SUBDIR


# ---- Dataset-aware dispatch wrappers (Type B) -----------------------------


def _require_supported(dataset: str) -> str:
    """Return the canonical benchmark name, raising for unsupported datasets.

    The tier-2 FL output dirs (``FL/Ochiai``, ``FL/SBIR``, ``FL/BoostN``) share the
    same subdir names across benchmarks, so the only per-dataset variation is the
    processed root. This guard keeps an unknown benchmark from silently resolving
    to ``data/<name>/processed`` (mirrors the D4J/BIP checks in ``get_src_dir``).
    """
    canonical = normalize_benchmark_name(dataset)
    if canonical not in ("D4J", "BIP"):
        raise NotImplementedError(f"Dataset {dataset!r} is not supported yet.")
    return canonical


def get_processed_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return processed-data dir for a given bug; dispatches on ``dataset``."""
    _require_supported(dataset)
    return LAYOUT.processed_root(dataset) / project / str(bug_id)


def get_ochiai_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    d = get_processed_dir(project, bug_id, dataset) / OCHIAI_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_boostn_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    d = get_processed_dir(project, bug_id, dataset) / BOOSTN_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sbir_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    d = get_processed_dir(project, bug_id, dataset) / SBIR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_stopwords_file(dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return the stopwords list for *dataset* (Java for D4J, Python for BugsInPy)."""
    canonical = _require_supported(dataset)
    return STOPWORDS_FILE_PYTHON if canonical == "BIP" else STOPWORDS_FILE


def get_sr_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return processed_dir/FlexFL/SR/ for Agent4SR corpus and model outputs.

    Benchmark-generic (tier-2 CEFL output): the ``FlexFL/SR`` subdir name is shared
    across benchmarks; only the processed root varies by ``dataset``.
    """
    d = get_processed_dir(project, bug_id, dataset) / FLEXFL_SR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sr_model_dir(
    project: str,
    bug_id: str | int,
    model: str,
    dataset: str = DEFAULT_BENCHMARK,
    *,
    model_id: str | None = None,
) -> Path:
    """Return processed_dir/FlexFL/SR/Agent4SR/<dir_name>/ for a specific model's outputs."""
    name = model_id if model_id is not None else _model_slug(model)
    d = get_sr_dir(project, bug_id, dataset) / "Agent4SR" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rankings_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return processed_dir/FlexFL/SR/rankings/ for standardised ranking outputs."""
    d = get_sr_dir(project, bug_id, dataset) / "rankings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lr_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    """Return processed_dir/FlexFL/LR/ for Agent4LR per-model outputs."""
    d = get_processed_dir(project, bug_id, dataset) / FLEXFL_LR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lr_model_dir(
    project: str,
    bug_id: str | int,
    model: str,
    dataset: str = DEFAULT_BENCHMARK,
    *,
    lr_model_id: str | None = None,
) -> Path:
    """Return processed_dir/FlexFL/LR/Agent4LR/<dir_name>/ for an LR model's outputs."""
    name = lr_model_id if lr_model_id is not None else _model_slug(model)
    d = get_lr_dir(project, bug_id, dataset) / "Agent4LR" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lr_candidate_file(
    project: str,
    bug_id: str | int,
    *,
    sr_model_id: str,
    dataset: str = DEFAULT_BENCHMARK,
) -> Path:
    """Return ``rankings/top20/<sr_model_id>.txt`` for the given SR model.

    This is the file consumed by :func:`src.agent4lr.io.load_lr_bug_inputs`
    as Agent4LR's candidate list. Does not check that the file exists.
    """
    return get_rankings_dir(project, bug_id, dataset) / "top20" / f"{sr_model_id}.txt"


def get_lr_checkpoints_dir(
    project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK
) -> Path:
    """Return processed_dir/FlexFL/LR/checkpoints/ for the content-addressable store."""
    d = get_lr_dir(project, bug_id, dataset) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lr_batch_registry_dir() -> Path:
    """Return ``PROJECT_ROOT/data/batches/agent4lr`` for the OpenAI batch registry.

    Deliberately anchored on ``PROJECT_ROOT`` (never relocated via
    ``CEFL_D4J_WORKSPACE``/``CEFL_BIP_WORKSPACE``) so batch bookkeeping
    and archived request/response JSONLs survive scratch wipes. Does not
    create the directory — the registry creates it lazily on first write.
    """
    return PROJECT_ROOT / "data" / "batches" / "agent4lr"


def get_repo_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    _require_d4j(dataset)
    return get_repo_dir_d4j(project, bug_id)


def get_src_dir(project: str, bug_id: str | int, dataset: str = DEFAULT_BENCHMARK) -> Path:
    canonical = normalize_benchmark_name(dataset)
    if canonical == "D4J":
        return get_src_dir_d4j(project, bug_id)
    if canonical == "BIP":
        return get_src_dir_bugsinpy(project, bug_id)
    raise NotImplementedError(f"Dataset {dataset!r} is not supported yet.")
