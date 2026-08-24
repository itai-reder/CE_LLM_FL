# CE-LLM-FL — Cost-Effective LLM Fault Localization

A reproducible pipeline for studying **where the money goes** in LLM-driven fault localization.
Five fault-localization methods — three classical, two LLM-based — run over the same two
benchmarks, with per-configuration token and cost accounting attached to every metric.

The research question this repository exists to answer: an LLM agent chain for fault localization
has several slots, each of which can be filled by a different model tier and reasoning effort.
Which combinations buy accuracy, and which only buy tokens?

| | |
|---|---|
| **Benchmarks** | Defects4J (Java, 854 bugs / 17 projects) · BugsInPy (Python, 501 bugs / 17 projects) |
| **Classical FL** | Ochiai (SBFL) · SBIR (SBFL + IR, rank-aggregated) · BoostN (method-level BM25) |
| **LLM FL** | Agent4SR (search-and-retrieval) · Agent4LR (learning-to-rank), 3-slot chains |
| **Grid** | 72 Agent4LR configurations on Defects4J, 41 on BugsInPy |
| **Evaluated** | 854 / 390 bugs with classical baselines · 626 / 288 bugs with the LLM grid |
| **Metrics** | First Rank, Average Rank, Top-1…Top-5, Wasted Effort, input/cached/output tokens, USD |

`CEFL` appears throughout the code and tooling as the project's short name (**C**ost-**E**ffective
LLM **F**ault **L**ocalization) — in environment variables (`CEFL_*`), Docker image tags
(`defects4j:cefl`) and container names.

---

## Reproducing the results without any benchmark data

Every number in the paper is computed from `results/` and `evaluation/`, both committed here.
`results/` holds the slimmed per-bug inputs — candidate universe, rankings, ground truth, and each
Agent4LR configuration's top-5 with its token usage — and nothing else. Re-deriving the tables
therefore needs no benchmark checkout, no containers, no API keys, and no downloads:

```bash
conda env create -f environment.yml     # creates ce_llm_fl, installs requirements.txt
conda activate ce_llm_fl
export PYTHONPATH=fl_methods            # required for the `from src…` imports

python fl_methods/run_evaluation.py --benchmark defects4j --all-projects --force
python fl_methods/run_evaluation.py --benchmark bugsinpy  --all-projects --force

git diff --stat evaluation/ results/    # expected: no changes
```

The evaluator reads `results/` exclusively and rewrites the same files, so an unmodified checkout
reproduces byte-identical output. That is the reproducibility claim in its strongest form: the
committed CSVs are not a snapshot you have to trust, they are a fixed point you can recompute.

`results/` is not an abridged summary. It is the exact input closure the metrics depend on, and
`run_build_results.py` refuses to write a bug's directory unless the metrics computed from it match
the metrics computed from the full processed tree, for every SR model — see
[`docs/data_contract.md` §7](docs/data_contract.md).

### What the outputs mean

| Path | Contents |
|---|---|
| `evaluation/<BM>/{baselines,flexfl}.csv` | one row per (bug, method/config) — FR, AR, Top-K, WE, tokens, cost |
| `evaluation/<BM>/*_first.csv` | the same, scored against faults reachable from the *first* triggering test |
| `evaluation/<BM>/*_summary.csv` | one row per method/config, averaged across bugs |
| `evaluation/BIP/exclusions.csv` | why each non-evaluable BugsInPy bug was excluded, bucketed |
| `results/<BM>/<Project>/<Bug>/evaluation/` | the same four metric files, per bug |

Column semantics are in [`docs/data_contract.md` §8](docs/data_contract.md).

---

## Repository layout

```
fl_methods/            pipeline code; run_*.py are the CLI entry points
  src/extraction/      benchmark data extraction (Docker-wrapped)
  src/sbir/            Ochiai parsing, Blues IR, RAFL rank aggregation
  src/boostn/          method-level BM25 with adaptive k1
  src/agent4sr/        SR agent loop, per-bug corpus, 7 lookup tools
  src/agent4lr/        LR 3-slot chain, providers, checkpoints, OpenAI Batch
  src/results/         processed/ → results/ builder with equivalence self-check
  src/evaluation/      metrics, per-bug + cross-bug aggregation, exclusions
  src/common/ src/core/ src/benchmarks/   paths, parsers, layout, adapters
results/               COMMITTED slim per-bug evaluation inputs
evaluation/            COMMITTED cross-bug metric tables
tests/                 pytest suite (883 tests)
utils/docker/          image build + container lifecycle scripts per benchmark
utils/build_gzoltar.sh GZoltar JAR build
docs/data_contract.md  the on-disk schema contract — start here to extend a benchmark
data/ transcripts/ logs/   gitignored: benchmark data, full LLM transcripts, run logs
```

Per-package details are in the module docstrings; `docs/data_contract.md` is the authoritative
description of every file the pipeline reads or writes.

---

## Full pipeline

Only needed to regenerate data from scratch — for the published metrics, see the section above.

### 1. Environment

```bash
conda env create -f environment.yml && conda activate ce_llm_fl
pip install -r requirements-dev.txt      # adds ruff, mypy, pytest
export PYTHONPATH=fl_methods
```

### 2. Containers

Each benchmark runs its builds and tests inside its own image. The setup scripts clone the
upstream benchmark, build the image, and start a long-lived container with `data/<BM>/` bound at
`/workspace`:

```bash
bash utils/docker/defects4j/setup-docker4j.sh     # defects4j:cefl  (JDK 11 + defects4j CLI)
bash utils/docker/bugsinpy/setup-docker-bip.sh    # bugsinpy:cefl   (per-project envs + FauxPy)
bash utils/build_gzoltar.sh                       # GZoltar JARs → utils/java/
```

`start-*.sh` / `stop-*.sh` control the containers; `cmd-*.sh` runs a one-shot command inside one.

### 3. Extraction

```bash
python fl_methods/run_extraction.py -p Csv -v 2                        # Defects4J
python fl_methods/run_extraction.py --benchmark bugsinpy -p thefuck -v 1
python fl_methods/audit_bip_extraction.py --projects thefuck           # BugsInPy health report
```

Omit `-v` to process every bug of a project. Produces `data/<BM>/processed/<Project>/<Bug>/`:
properties, test lists, `method_signatures.csv`, coverage + an Ochiai ranking, fault ground truth,
and the bug report.

### 4. Classical fault localization

```bash
python fl_methods/run_sbir.py   -p Csv -v 2      # Ochiai + Blues + SBIR
python fl_methods/run_boostn.py -p Csv -v 2      # BoostN
```

### 5. LLM stages

```bash
python fl_methods/run_agent4sr.py corpus  -p Csv -v 2     # per-bug method corpus
python fl_methods/run_agent4sr.py run     -p Csv -v 2 --model llama3.1:8b
python fl_methods/run_agent4sr.py combine -p Csv -v 2     # standardized rankings + top-20

python fl_methods/run_lr.py       -p Csv -v 2 --config M1R1-M1R1-M1R1
python fl_methods/run_lr_batch.py --experiment swap-expanded --auto   # OpenAI Batch API sweep
```

Agent4SR talks to Ollama (`OLLAMA_BASE_URL`, default `http://localhost:11434`). Agent4LR uses the
OpenAI Responses API or Ollama depending on the configuration in
`fl_methods/configs/lr_configs.json`; OpenAI runs need `OPENAI_API_KEY`.

### 6. Build `results/` and evaluate

```bash
python fl_methods/run_build_results.py --benchmark defects4j --all-projects
python fl_methods/run_evaluation.py    --benchmark defects4j --all-projects
```

`run_build_results.py` distils each processed bug into `results/`, copies the full LLM records into
the gitignored `transcripts/` tree, and verifies that the slim directory yields exactly the same
metrics as the full one before keeping it.

---

## Data

Processed benchmark data is too large to track in git (≈100 GB with transcripts), so `data/` is
gitignored. The per-project archives are published on Google Drive:

**<https://drive.google.com/drive/folders/1G9EJehTiT88eQFqTegZkUrHXMHlS4iOe>**

Layout: `{D4J,BIP}/<Project>.zip`, each archive expanding to `<Project>/<BugId>/<processed tree>`.
Unzip into `data/<BM>/processed/` to place a project where the pipeline expects it. The archives
are produced by:

```bash
python utils/package_drive.py --benchmark d4j -p Lang        # → drive_packages/D4J/Lang.zip
python utils/package_drive.py --benchmark bip --all-projects
```

Coverage binaries (`coverage.ser`, `matrix.txt`) and raw FauxPy reports are excluded — they are
large and regenerable.

Downloading data is **not** required to reproduce the published metrics; it is required to re-run
extraction, the classical methods, or the LLM stages on your own machine.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PYTHONPATH` | — | must include `fl_methods` |
| `CEFL_D4J_WORKSPACE` | `data/D4J` | host directory bound at `/workspace` in the Defects4J container |
| `CEFL_BIP_WORKSPACE` | `data/BIP` | the same for BugsInPy |
| `CEFL_D4J_CONTAINER` | `defects4j-cefl-container` | container name to exec into |
| `CEFL_BIP_CONTAINER` | `bugsinpy-cefl-container` | the same for BugsInPy |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Agent4SR / Ollama-backed Agent4LR endpoint |
| `OPENAI_API_KEY` | — | required for OpenAI-backed Agent4LR configurations |

A GitHub token in `.gh_token` (gitignored) raises the rate limit for bug-report fetching.

---

## Development

```bash
pytest                                # full suite
ruff check . && ruff format --check .  # lint + format
mypy .                                # type check
```

Conventions: absolute `from src…` imports, `pathlib.Path` for filesystem access, module-level
`logging.getLogger(__name__)`, dataclasses for structured payloads. Configuration lives in
`pyproject.toml` (ruff line length 100, mypy 3.11, pytest rooted at `tests/`).

---

## Analysis

The research-question notebooks live on the **`results_analysis`** branch, which builds its frames
directly from `evaluation/`:

```bash
git checkout results_analysis
jupyter lab analysis/research_questions.ipynb
```

Keeping them off `main` keeps the pipeline and its committed results independent of any particular
presentation of them.
