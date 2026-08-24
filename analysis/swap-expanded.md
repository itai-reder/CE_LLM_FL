# Experiment: `swap-expanded`

A grid generalization of the prior single-role swap experiments
(`planner_swap`, `tool_caller_swap`, `finisher_swap`). Instead of swapping one role
against a single `M2R1` anchor, `swap-expanded` sweeps every swapped variant against
three different fixed background tiers (`i ∈ {1,2,3}`), for each of the three Agent4LR
roles (Planner, Tool-caller, Finisher), on the full Defects4J benchmark.

**Cut grid (39 configs).** The full grid would sweep every `M<m>R<r>` with
`m,r ∈ {1,2,3}` (75 unique configs). To keep the run tractable, the swept variant is
**pruned to `m == 2` OR `r == 1`**: only the mid model `M2` (gpt-5-mini) gets its full
reasoning sweep `R1..R3`; the cheap/expensive models `M1` (gpt-5-nano) and `M3` (gpt-5)
are pinned at `R1`. This yields **5 swept variants per (role, i)** —
`{M1R1, M2R1, M2R2, M2R3, M3R1}` — and **39 unique configs** in total. The pruning lives
in `_swap_expanded_configs()` (`fl_methods/run_experiment.py`, the
`if m != 2 and r != 1: continue` guard).

## Configuration naming

Configs use a compact human-facing label `<X><i>(M<m>R<r>)` with `i,m,r ∈ {1,2,3}`
and `X ∈ {P,T,F}` (subject to the cut above: `(m,r)` is restricted to `m == 2` or
`r == 1`). Each maps to the repo's canonical
`Planner-ToolCaller-Finisher` config string (the keys in
`fl_methods/configs/lr_configs.json`):

| Label form | X | Meaning | Canonical config string |
|---|---|---|---|
| `P<i>(M<m>R<r>)` | P | Planner swapped, other two roles fixed at `M<i>R1` | `M<m>R<r>-M<i>R1-M<i>R1` |
| `T<i>(M<m>R<r>)` | T | Tool-caller swapped, other two roles fixed at `M<i>R1` | `M<i>R1-M<m>R<r>-M<i>R1` |
| `F<i>(M<m>R<r>)` | F | Finisher swapped, other two roles fixed at `M<i>R1` | `M<i>R1-M<i>R1-M<m>R<r>` |

Tier decode (authoritative in `lr_configs.json`):

- **Model (M):** `M1` = gpt-5-nano, `M2` = gpt-5-mini, `M3` = gpt-5 (all OpenAI).
- **Reasoning (R):** `R1` = minimal, `R2` = low, `R3` = medium. (Only up to `R3`, since `r ∈ {1,2,3}`.)

**Grid size.** After the cut, the swept variant ranges over 5 values
`{M1R1, M2R1, M2R2, M2R3, M3R1}`, so `3 X × 3 i × 5 = 45` label instances collapse to
**39 unique configs**: the homogeneous anchors `M<i>R1-M<i>R1-M<i>R1` are produced once
per role (`P<i>(M<i>R1) = T<i>(M<i>R1) = F<i>(M<i>R1)`), so the three anchors are shared
(45 − 6 = 39). The experiment definition (`_swap_expanded_configs()` in
`fl_methods/run_experiment.py`) enumerates and de-duplicates them.

(The un-cut full grid would be `3 X × 3 i × 3 m × 3 r = 81` instances → 75 unique
configs; those 75 keys still exist in `lr_configs.json`, but only these 39 are wired into
the `swap-expanded` experiment.)

### Full config listing

Swept variant ranges over the 5 cut values `{M1R1, M2R1, M2R2, M2R3, M3R1}` (only `M2`
gets `R2`/`R3`; `M1`/`M3` are pinned at `R1`).

### P — Planner swapped

| Fixed tier i | Swept configs (swap ∈ {M1R1, M2R1, M2R2, M2R3, M3R1}) |
|---|---|
| i=1 | M1R1-M1R1-M1R1 *(anchor)*, M2R1-M1R1-M1R1, M2R2-M1R1-M1R1, M2R3-M1R1-M1R1, M3R1-M1R1-M1R1 |
| i=2 | M1R1-M2R1-M2R1, M2R1-M2R1-M2R1 *(anchor)*, M2R2-M2R1-M2R1, M2R3-M2R1-M2R1, M3R1-M2R1-M2R1 |
| i=3 | M1R1-M3R1-M3R1, M2R1-M3R1-M3R1, M2R2-M3R1-M3R1, M2R3-M3R1-M3R1, M3R1-M3R1-M3R1 *(anchor)* |

### T — Tool-caller swapped

| Fixed tier i | Swept configs (swap ∈ {M1R1, M2R1, M2R2, M2R3, M3R1}) |
|---|---|
| i=1 | M1R1-M1R1-M1R1 *(anchor)*, M1R1-M2R1-M1R1, M1R1-M2R2-M1R1, M1R1-M2R3-M1R1, M1R1-M3R1-M1R1 |
| i=2 | M2R1-M1R1-M2R1, M2R1-M2R1-M2R1 *(anchor)*, M2R1-M2R2-M2R1, M2R1-M2R3-M2R1, M2R1-M3R1-M2R1 |
| i=3 | M3R1-M1R1-M3R1, M3R1-M2R1-M3R1, M3R1-M2R2-M3R1, M3R1-M2R3-M3R1, M3R1-M3R1-M3R1 *(anchor)* |

### F — Finisher swapped

| Fixed tier i | Swept configs (swap ∈ {M1R1, M2R1, M2R2, M2R3, M3R1}) |
|---|---|
| i=1 | M1R1-M1R1-M1R1 *(anchor)*, M1R1-M1R1-M2R1, M1R1-M1R1-M2R2, M1R1-M1R1-M2R3, M1R1-M1R1-M3R1 |
| i=2 | M2R1-M2R1-M1R1, M2R1-M2R1-M2R1 *(anchor)*, M2R1-M2R1-M2R2, M2R1-M2R1-M2R3, M2R1-M2R1-M3R1 |
| i=3 | M3R1-M3R1-M1R1, M3R1-M3R1-M2R1, M3R1-M3R1-M2R2, M3R1-M3R1-M2R3, M3R1-M3R1-M3R1 *(anchor)* |

The three anchors (`M1R1-M1R1-M1R1`, `M2R1-M2R1-M2R1`, `M3R1-M3R1-M3R1`) appear once in
each of the P/T/F blocks above but are a single config each — 39 unique in total.

---

## 1. Run the full FlexFL pipeline

> **Cost warning.** 39 configs × the full Defects4J benchmark, many using gpt-5, is a
> large and expensive run. Validate first with `--dry-run` (and/or a single `-p Lang -n 1`)
> before launching the full sweep.

**Prerequisites** (from repo root):

```bash
conda activate ce_llm_fl
export PYTHONPATH=fl_methods
export OPENAI_API_KEY=…          # required for the gpt-5-family configs
# Defects4J container running + GZoltar JARs built (see README / utils/docker/defects4j/).
```

**Primary command** — the full benchmark, all 39 configs:

```bash
PYTHONPATH=fl_methods python fl_methods/run_experiment.py --experiment swap-expanded
```

(Point `--sr-base-url` at your Ollama endpoint if it is not the local default;
add `--no-sr-verify` for self-signed TLS.)

`--experiment swap-expanded` defaults to `projects=None` (all D4J projects) and the full
benchmark (`n_bugs_per_project = FULL_BENCHMARK`, all bugs per project).

**Relevant optional flags** (`fl_methods/run_experiment.py`):

- `-p, --projects <P …>` — restrict to specific projects (default: all).
- `-n, --n-bugs-per-project <int>` — cap bugs per project (default: full benchmark). Use
  `-p Lang -n 1` for a smoke test.
- `--lr-configs <cfg …>` — run only a subset of the 39 configs (names must exist in
  `lr_configs.json`).
- `--skip-stage <stage …>` / `--force-stage <stage …>` — stage set is
  `extraction sbir boostn agent4sr agent4lr evaluation`. Skip already-done upstream stages
  (e.g. `--skip-stage extraction sbir boostn agent4sr`) to only (re)run the LR sweep +
  evaluation; force to re-run despite existing outputs.
- `--workers <int>` — parallelize across bugs (LR configs still run serially within a bug,
  because they share the per-bug checkpoint cache — see below).
- `--dry-run` — print the per-stage subprocess argv without executing. Always validate here first.
- `--log-dir <path>` — run log/report location (default `logs/D4J/experiment/`).
- `--sr-model <slug>` — SR model (default `llama3.1:8b`). **Keep the default:** the
  orchestrator's `agent4lr` subprocess does not forward `--sr-model-id`, so LR looks for
  `FlexFL/SR/rankings/top20/llama3.1_8b.txt`. Changing `--sr-model` without matching that
  slug will break LR candidate lookup.
- SR transport: `--sr-base-url`, `--no-sr-verify` (self-signed BGU Ollama).

**Checkpoint cache.** Per-bug LLM responses are cached under
`data/D4J/processed/<Project>/<BugId>/FlexFL/LR/checkpoints/` and reused across configs, so
the 39 LR configs run serially within a bug (parallelism is across bugs via `--workers`).

**Outputs per bug** (`data/D4J/processed/<Project>/<BugId>/`):

- `FlexFL/LR/Agent4LR/<config>/lr_result.json` (+ `top5.txt`) — one dir per config.
- `FlexFL/SR/rankings/top5/<config>.{txt,csv}` — final top-5 per config.
- Run logs/report under `logs/D4J/experiment/`.

### Resume an interrupted / partially-failed run

Agent4LR is **skip-on-existing** at the granularity of one `(bug, config)`: a config is
re-run unless **both** `lr_result.json` and `top5.txt` already exist for it
(`run_agent4lr_for_bug` returns `SKIPPED`). So simply **re-launching the experiment
resumes it** — completed configs are skipped for free (no tokens spent), and only the
missing/failed ones execute. (`run_lr.py --force` is a no-op; there is intentionally no
CLI way to force a redo of completed configs.)

To resume just the LR sweep (upstream stages already done) plus evaluation:

```bash
export OPENAI_API_KEY=…          # a mass agent4lr failure is almost always an unset key
                                 # or an exhausted credit balance — fix that first
PYTHONPATH=fl_methods python fl_methods/run_experiment.py \
    --experiment swap-expanded \
    --skip-stage extraction sbir boostn agent4sr
```

Notes:

- **Do not** pass `--force-stage agent4lr` — it only defeats the skip and re-burns tokens
  on already-completed configs.
- Bugs with **no faulty method in the SR top-20** always skip (they write nothing, by
  design — see `_any_fault_in_candidates`); expected, not a failure.
- The `--skip-stage …` above assumes upstream artifacts exist. A bug whose upstream stage
  failed (missing SR `top20`, corpus, etc.) will make its LR configs fail with
  `FileNotFoundError`; repair those by re-running the full pipeline for just those bugs
  (drop `--skip-stage`, add `-p <Project> -n <…>`).
- Preview what will run with `--dry-run`; narrow with `-p` / `--lr-configs` for a subset.

## 2. Commit the new state after running

The sweep writes into the gitignored `data/D4J` tree. What gets committed is the slim
`results/` layer distilled from it (with the builder's metric-equivalence self-check)
plus the regenerated cross-bug tables:

```bash
PYTHONPATH=fl_methods python fl_methods/run_build_results.py -p <Project> --force
git add results/D4J evaluation/D4J
git commit -m "data(swap-expanded): results + evaluation for the swap-expanded sweep"
```

## 3. Evaluate — local + global

Evaluation reads the slim `results/` tree (build it first with `run_build_results.py`)
and auto-discovers configs from each bug's `lr.json`; no registry edit is needed. The
scoring universe is the SR top-20 (`rankings/top20/llama3.1_8b.txt`), shared by the
baselines and all configs.

```bash
# Per project (repeat per project, or loop over all D4J projects)
PYTHONPATH=fl_methods python fl_methods/run_evaluation.py -p <Project> --force
```

- **Local (per-project / per-bug):** writes
  `results/D4J/<Project>/<BugId>/evaluation/{baselines,baselines_first,flexfl,flexfl_first}.csv`
  — one row per FL method (baselines = Ochiai/SBIR/BoostN; flexfl = the Agent4LR configs).
  Use `--require-configs <config …>` to only evaluate bugs where every named config has a
  valid `top5`.
- **Global (benchmark-wide):** the same run appends to / regenerates the cross-bug tables in
  `evaluation/D4J/`:
  - `flexfl.csv` / `flexfl_first.csv` — long form, one row per Project×BugId×Method.
  - `flexfl_summary.csv` / `flexfl_first_summary.csv` — per-Method aggregates across all D4J
    bugs: MFR, MAR, TopKRate (Top1..Top5), MeanWE, token totals, MeanCostUSD.

  `_first` variants score against the first-triggering-test ground truth only; non-`_first`
  against all faults. Bugs with no faulty method in their universe (blank FR) are dropped
  from the summary aggregates rather than counted as failures.

Commit the evaluation CSVs the same way as step 2 (they live under `evaluation/D4J/`).
