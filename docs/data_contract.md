# Data Contract

This document defines the **files on disk** that the pipeline reads and writes. It exists so that
adding a benchmark beyond Defects4J (Java) and BugsInPy (Python) is a checklist of
files-to-produce rather than a reverse-engineering exercise, and so that anyone reproducing the
published numbers knows exactly what each artifact means.

The contract is the **schema**, not the tool that produces it. A benchmark using `coverage.py`
and `ast` is free to skip GZoltar and `javalang` entirely as long as it lands the same shapes at
the paths `src/common/config.py` resolves.

Running examples come from `Lang/8` (Defects4J, `FastDateFormat` timezone bug) and `thefuck/2`
(BugsInPy). Where a bug lacks a particular output, another is substituted and named.

---

## 1. Terminology

- **Bug** — a historical snapshot of a project, e.g. `Lang/1`. Some benchmarks call this a *case*.
- **Buggy / fixed version** — the two snapshots a bug ships, one commit apart. "Buggy" unless said otherwise.
- **Repos / fixed directory** — where snapshots are checked out: `data/<BM>/repos/<Project>/<BugId>/`, `data/<BM>/fixed/...`.
- **Processed directory** — everything extracted or derived for a bug: `data/<BM>/processed/<Project>/<BugId>/`. The canonical handoff surface between stages.
- **Results directory** — the committed, slimmed evaluation inputs for a bug: `results/<BM>/<Project>/<BugId>/` (§7).
- **Triggering test** — fails on the buggy version, passes on the fixed one. The **first** triggering test is the first entry of the benchmark's trigger list; if the benchmark defines no order, the adapter must.
- **Component / signature** — a source entity at some granularity (statement, method, class) and its unique ID. Method-level by default.
- **Candidate** — a component covered by at least one triggering test. **First-only candidate**: covered by the first triggering test.
- **Fault** — a component modified between buggy and fixed *and* covered by a triggering test. Diff lines outside any known component are ignored.
- **Corpus** — the per-bug method index used by the IR- and LLM-based methods, stored line-aligned as `corpus_methods.txt` + `corpus_codes.txt`.
- **SBFL / IRFL** — spectrum-based (Ochiai, SBIR) and IR-based (BoostN, SBIR) fault localization.
- **SR / LR** — the two LLM stages: Agent4SR *search-and-retrieval* (ends with a top-5 and a top-20 candidate list) and Agent4LR *learning-to-rank* (re-ranks that top-20 down to its own top-5).
- **FR / AR / Top-K / WE** — evaluation metrics: First Rank, Average Rank, Top-K success (1 iff FR ≤ K), Wasted Effort.

---

## 2. Pipeline overview

```
┌─────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐
│ §3      │→ │ §4         │→ │ §5       │→ │ §5       │→ │ §6       │→ │ §7       │→ │ §8        │
│ Setup   │  │ Extraction │  │ Ochiai   │  │ Agent4SR │  │ Agent4LR │  │ results/ │  │ Evaluation│
│container│  │ processed/ │  │ BoostN   │  │ top-5 +  │  │ re-ranks │  │ slim,    │  │ per-bug + │
│ + utils │  │ per bug    │  │ SBIR     │  │ top-20   │  │ to top-5 │  │ committed│  │ cross-bug │
└─────────┘  └────────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └───────────┘
```

Stages §4–§6 read and write only the processed directory (plus the buggy checkout for source).
Handoffs are file paths; there is no in-memory IPC. Stage §7 distils the processed tree into the
committed `results/` tree, and §8 evaluates **from `results/` alone** — so the published numbers
can be reproduced with no benchmark data, no containers, and no API keys.

Each section below states its **prereqs**, **operations**, **deliverables** (with schemas) and
**consumers**.

---

## 3. Stage 0 — Setup

### Prereqs
A container exposing the benchmark's build/test toolchain. Defects4J ships JDK 11 + the
`defects4j` CLI; BugsInPy ships per-project conda environments + FauxPy.

### Operations
`src/extraction/docker.py` is the single runtime backend. Its public surface:

| Symbol | Role |
|---|---|
| `docker_exec(cmd, cwd, user, check)` | Run a command inside the container. The only mandatory primitive. |
| `run_defects4j(args, ...)` | Convenience wrapper for `defects4j <args>`. *Defects4J-specific.* |
| `run_java_in_container(java_args, ...)` | In-container `java` invocation (GZoltar). *Java-specific.* |
| `to_container_path(host_path)` | Translate host paths under `data/<BM>/` to `/workspace/...`. |
| `translate_cmd_paths(cmd)` | Apply `to_container_path` to every absolute argument. |
| `ensure_java_utils_in_workspace()` | Stage the GZoltar JARs from `utils/java/` into `data/D4J/deps/`. *Java-specific.* |
| `is_container_running()`, `D4J_JUNIT_CONTAINER` | Liveness probe; canonical JUnit jar path. |

BugsInPy commands go through `src/extraction/bugsinpy.py:run_bip`, which drives the same
container primitives via the wrapper script staged inside the image.

### Deliverables
Setup is a capability, not a file:

- A running container — `defects4j-cefl-container` / `bugsinpy-cefl-container` by default,
  overridable with `CEFL_D4J_CONTAINER` / `CEFL_BIP_CONTAINER`.
- A workspace bind-mounted at `/workspace` — `data/D4J/` and `data/BIP/` by default,
  relocatable with `CEFL_D4J_WORKSPACE` / `CEFL_BIP_WORKSPACE`.
- Helper binaries staged: for Defects4J, `utils/java/gzoltar-{agent,cli}.jar` copied into
  `data/D4J/deps/`.

Host-side scripts live in `utils/docker/<benchmark>/` (`setup-*.sh`, `start-*.sh`, `stop-*.sh`,
`cmd-*.sh`). A new benchmark should mirror that layout.

---

## 4. Stage 1 — Extraction

CLI: `fl_methods/run_extraction.py`. The canonical step order is in `src/extraction/pipeline.py`:

```python
ALL_STEPS = ("repo_setup", "signatures", "tests", "gzoltar", "faults", "bug_report")
```

For each bug: check out buggy → compile → export properties → enumerate tests → extract method
signatures → run coverage and produce an Ochiai ranking → diff buggy against fixed → fetch the
bug report.

All paths below are relative to `data/<BM>/processed/<Project>/<BugId>/`.

### 4.1 Project metadata

Plain-text files, one value or one value per line. Defects4J writes them via
`defects4j export -p <prop>` (`src/extraction/d4j.py:EXPORT_PROPERTIES`); BugsInPy sources the
same *values* from `bugsinpy_bug.info` and does not fabricate Java-named property files.

| File | Schema | Lang/8 sample |
|---|---|---|
| `dir.src.classes` | relative path | `src/main/java` |
| `dir.bin.classes` | relative path | `target/classes` |
| `dir.src.tests` | relative path | `src/test/java` |
| `dir.bin.tests` | relative path | compiled test root |
| `cp.test` | colon-joined classpath | consumed by GZoltar |
| `classes.modified` | FQCN per line | `org.apache.commons.lang3.time.FastDatePrinter` |
| `classes.relevant` | FQCN per line | classes touched by triggering tests |
| `tests.relevant` | test FQCN per line | `org.apache.commons.lang3.time.DateFormatUtilsTest` |
| `tests.trigger` | `Class::method` per line | `…FastDateFormat_PrinterTest::testCalendarTimezoneRespected` |

These names carry Defects4J provenance. The **schema** is the contract; the filenames are not.
Most consumers read them through `Repo` accessors and `src/common/config.py` helpers rather than
by literal name — a new benchmark should read its own native metadata and let `config.py` resolve
where it lives.

**Consumed by:** every downstream stage.

### 4.2 Test enumeration

| File | Format | Notes |
|---|---|---|
| `all_tests.txt` | `Class#method` per line | Every test method the runner can discover. |
| `relevant_tests.txt` | `Class#method` per line | Subset exercising `classes.relevant`. |
| `junit_tests.txt` | one test class per line | GZoltar input format. |
| `failing_tests` | raw runner dump | Diagnostics. |
| `failing_tests.txt` | `Class::method` per line | Parsed failures. |
| `trigger_tests` | raw trigger dump (header + stack) | Source for the first-test prompt. |
| `trigger_test_clean.txt` | trigger test source + cleaned stack trace | Written by `src/extraction/trigger_test.py`. Optional; may be empty. |

Both `Class#method` and `Class::method` appear in the pipeline and both are load-bearing — the
hash form is GZoltar's convention, the double-colon form is Defects4J's. Downstream code
dispatches on the format.

**Consumed by:** coverage (`junit_tests.txt`, `relevant_tests.txt`), Agent4SR and Agent4LR prompts
(`trigger_test_clean.txt`).

### 4.3 Method corpus — `method_signatures.csv`

The canonical mapping from (file path, line) to method. Every method-level computation keys off
it. Written by `run_corpus_method_extraction` in `src/extraction/gzoltar.py`.

Schema — `corpus_id;path;startLine;endLine`, semicolon-delimited:

```
corpus_id;path;startLine;endLine
org.apache.commons.lang3$AnnotationUtils.getShortClassName(java.lang.Class<?>);org/apache/commons/lang3/AnnotationUtils.java;68;85
```

- `corpus_id` has the shape `pkg$Class.method(SimpleParams)` — `$` separates package from class (§9).
- `path` is relative to the source root (i.e. starts inside `dir.src.classes`).
- `startLine` / `endLine` are 1-based inclusive line numbers in the buggy file.

Row order matters: the exact-match index built in `src/evaluation/sources.py` is first-wins and
the dotted-fallback index is last-wins, so any process that rewrites this file (§7) must preserve
the original order.

**Consumed by:** fault extraction, BoostN, statement→method aggregation
(`src/common/method_entity.py`), Agent4SR corpus generation, every `rankings/*.csv` writer.

### 4.4 Coverage

Defects4J writes GZoltar's four files under `sfl/sfl/txt/`; BugsInPy writes the same shapes under
`FauxPy/coverage/` and `FauxPy/reports/`. Paths are resolved through `config.py`, so a new
benchmark may use its own tool-native location — but the schemas below are required.

`spectra.csv` — statement IDs, one per line, in matrix-column order:

```
name
org.apache.commons.lang3.time$FastDateFormat#FastDateFormat(java.lang.String,java.util.TimeZone,java.util.Locale):368
```

`matrix.txt` — ASCII, space-separated. One row per test in `tests.csv` order; *N* binary columns
matching `spectra.csv`, then one outcome bit (0 = pass, 1 = fail).

`tests.csv` — comma-delimited test outcomes:

```
name,outcome,runtime,stacktrace
org.apache.commons.lang3.time.DateFormatUtilsTest#testConstructor,PASS,160252736,
```

`ochiai.ranking.csv` — **semicolon**-delimited, sorted descending:

```
name;suspiciousness_value
org.apache.commons.lang3.time$FastDateParser$TextStrategy#isNumber():570;1.0
```

**Consumed by:** the Ochiai parser (`src/sbir/sbfl.py`), SBIR aggregation (`src/sbir/rafl.py`),
and method aggregation via `method_signatures.csv`.

### 4.5 Faults — ground truth

Written by `src/extraction/faults.py`: the buggy checkout is diffed against the fixed checkout and
each changed line is resolved to its enclosing method through `method_signatures.csv`.

`faults.csv`:

```
path,line,signature
org/apache/commons/lang3/time/FastDatePrinter.java,1098,
```

Columns: source path relative to the source root, 1-based line in the buggy file, corpus id of the
enclosing method (empty when no method maps — logged as a warning).

`faults.txt` is the flat legacy form, `pkg.Class lineNum` (dotted class name, *not* the corpus `$`
form). `faults_first.csv` / `faults_first.txt` use the same schemas filtered to faults reachable
from the **first** triggering test; they are legitimately empty for some bugs (Lang/8 among them).

**Consumed by:** evaluation (`src/evaluation/per_bug.py`), and Agent4LR's
skip-if-no-recoverable-fault precheck.

### 4.6 Bug report — `bug_report.json`

Format-locked and identical across benchmarks; only the *sourcing* adapts (Defects4J queries
`defects4j query -q report.url`; BugsInPy traces `bug.info` → GitHub commit → issue).

```json
{
  "url": "https://issues.apache.org/jira/browse/LANG-818",
  "title": "FastDateFormat's \"z\" pattern does not respect timezone…",
  "raw": "The work on LANG-462 has introduced a time zone formatting bug…",
  "description": "The work on LANG-462 has introduced a time zone formatting bug…"
}
```

Required keys: `url`, `title`, `description`, `raw`; optional `error` if the fetch failed. Bodies
are fetched from JIRA / SourceForge / Google Code / GitHub by `src/extraction/report_parser.py`,
rate-limited per host. A new benchmark may ship cached JSON offline — no HTTP required.

**Consumed by:** BoostN (BM25 query), Blues, Agent4SR and Agent4LR prompts.

### Gating and validation

`run_extraction.py` exposes `--gzoltar-only`, `--bug-report-only`, `--faults-only`, `--force` and
`--no-validate`. `validate_extraction_outputs` checks that each expected file exists and is
non-empty; a run summary lands in `logs/<BM>/extraction/`.

---

## 5. Stage 2 — Traditional FL and Agent4SR

### 5.1 Ochiai / SBFL

**Prereqs:** the coverage-stage Ochiai ranking. **Operations:** `src/sbir/sbfl.py:parse_ochiai_csv`
converts `pkg$Class#method:line` IDs into statement IDs (`pkg.Class#line`).

Deliverables in `FL/Ochiai/`:

| File | Schema |
|---|---|
| `sbfl_ochiai.json` | `{"sbfl_scores": {"pkg.Class#line": float, …}}` |
| `stmt-susps.txt` | `Statement,Suspiciousness`, ranked descending |

### 5.2 BoostN

**Prereqs:** `bug_report.json`, the buggy source tree, `method_signatures.csv`, a stopwords file.
**Operations:** a per-corpus BM25 index with a custom variant (`src/boostn/boostn.py`) — adaptive
`k1` (0 for corpora ≤ 3000 methods, else 1), `b = 0.3`, IDF floored at `0.25 × avg_idf`; the query
is the preprocessed bug report.

Deliverables in `FL/BoostN/`: `boostn.json` (`{"boostn_scores": {corpus_id: float}}`) and
`boostn-method-susps.csv` (`Signature,Suspiciousness`, ranked descending).

### 5.3 SBIR (Blues + RAFL)

*Blues* runs BM25 over six per-class statement configurations, takes the consensus maximum, keeps
the top 100 and normalizes to [0, 1]. *RAFL* aggregates the Ochiai and Blues rankings by Borda
count, optionally refined by Cross-Entropy Monte Carlo over the Spearman footrule distance.

Deliverables in `FL/SBIR/`: `blues_scores.json`, `stmt-susps-blues.txt`, `sbir_scores.json`,
`sbir-susps.txt`.

### 5.4 Standardized rankings

`src/common/rankings.py` reduces the three baselines to method level (via
`aggregate_statement_scores_to_entities`) and writes them under `FlexFL/SR/rankings/`:

- `ochiai.csv`, `boostn.csv`, `sbir.csv` — `rank;signature;path;startLine;endLine;score`
- `top15.csv` / `top15.txt` — fused top-15 across the three baselines
- `top20/<sr_model_id>.{csv,txt}` — Agent4SR's top-20 candidates; **the input to Agent4LR**
- `top5/<lr_config>.{csv,txt}` — Agent4LR's per-config top-5, written after §6

The top-20 file uses **dotted** method IDs (`pkg.Class.method(P)`), not the corpus `$` form;
`src/agent4lr/io.py` canonicalizes on read.

### 5.5 Agent4SR

CLI: `fl_methods/run_agent4sr.py` (`corpus`, `run`, `combine`).

`corpus` walks the buggy source (excluding test directories) and writes three line-aligned files
into `FlexFL/SR/`: `corpus_methods.txt` (one corpus id per line), `corpus_codes.txt` (method
source, newlines flattened) and `corpus_to_sig.json` (dotted ID → corpus id).

`run` executes a ReAct loop against a chat endpoint (Ollama `/api/chat` by default, via
`OLLAMA_BASE_URL`): a planning turn, then up to `iterations` tool calls, then a finisher that
names five suspicious methods. The seven tools (`src/agent4sr/function_call.py`) are corpus-table
lookups — `get_paths`, `get_classes_of_path`, `get_methods_of_class`,
`get_code_snippet_of_method`, `find_class`, `find_method`, `exit` — not filesystem operations, so
they are language-agnostic by construction.

Deliverables in `FlexFL/SR/Agent4SR/<model_slug>/` (slug: `llama3.1:8b` → `llama3.1_8b`):
`sr_result.json` (full run record), `top5.txt`, `top5_raw.txt`, `top5_methods.txt`, and
`candidates/<model>_<input>/candidates.txt`.

---

## 6. Stage 3 — Agent4LR

CLI: `fl_methods/run_lr.py` (single bug) and `fl_methods/run_lr_batch.py` (OpenAI Batch API).

**Prereqs:** `FlexFL/SR/rankings/top20/<sr_model_id>.txt` (required); `bug_report.json` and
`trigger_test_clean.txt` (optional, selected by `--input`); `faults.csv` for the precheck.

**Operations:** a three-slot chain — `planner` → `tool_caller` → `finisher` — each a separate
model call, configured in `fl_methods/configs/lr_configs.json`. Config names in the published grid
have the form `M<tier>R<effort>-M<tier>R<effort>-M<tier>R<effort>`, one term per slot (e.g. `M1R1`
= `gpt-5-nano` at minimal reasoning).

- **Planner** — no tools (`tool_choice="none"`), free-form ranking plan.
- **Tool caller** — ReAct loop with `get_snippet_of_method(method_number)` (1-based index into the
  top-20) and `exit()`.
- **Finisher** — `tool_choice="required"`, must emit `rank_methods(top_5_methods: int[5])`, a
  permutation of top-20 indices.

Providers live in `src/agent4lr/providers/`: `openai` (Responses API, preserving reasoning items
across turns) and `ollama` (chat completions projected from the Responses shape). Checkpoints
(`src/agent4lr/checkpoints.py`) are content-addressed by (inputs descriptor, completed slot chain),
so partial chains resume.

Deliverables in `FlexFL/LR/Agent4LR/<config_name>/`: `lr_result.json`, `top5.txt`,
`checkpoint.json`. `lr_result.json` (schema_version 2) carries:

```
project, bug_id, config_name, agent_chain, sr_model_id, candidate_source,
started_at, finished_at, top5_indices, top5,
input_list, messages, response_dumps, input, schema_version
```

`input_list` is the canonical Responses-API conversation; `messages` is the readable transcript;
`response_dumps` holds the raw per-call provider payloads, including `usage.input_tokens`,
`usage.input_tokens_details.cached_tokens` and `usage.output_tokens`. Those feed the cost columns
via `src/evaluation/sources.py:load_agent4lr_usage` and `src/evaluation/pricing.py`.

---

## 7. Stage 4 — `results/`, the committed evaluation inputs

CLI: `fl_methods/run_build_results.py`. Implementation: `src/results/`.

A processed bug directory is large (hundreds of KB per bug, plus full LLM transcripts) and cannot
be committed for 1,355 bugs. `build_bug_results` distils each one into
`results/<BM>/<Project>/<BugId>/`, which carries **exactly** what evaluation reads:

```
results/<BM>/<Project>/<BugId>/
  method_signatures.csv          slimmed universe, same schema, source order preserved
  faults.csv  faults_first.csv   verbatim copies
  rankings/{ochiai,sbir,boostn}.csv   filtered to the resolution closure
  rankings/top20/<sr_model_id>.txt    verbatim copies
  lr.json                        every Agent4LR config: top5 + token usage + cost
  meta.json                      provenance + the exclusion signals BugsInPy needs
  evaluation/…                   per-bug metric CSVs (§8), written by run_evaluation.py
```

**The slimming rule.** The *resolution closure* is the set of entities the evaluation can reach
for this bug: everything resolvable from the SR top-20 lists, the fault signatures, and every
config's top-5 — resolved against the **full** corpus indexes at build time.
`method_signatures.csv` keeps exactly the ordered subsequence covering that closure (first row per
corpus id), so both lookup indexes resolve identically to the full file; the baseline ranking CSVs
keep only closure signatures, since rows outside the candidate universe cannot influence a metric.

**The guarantee.** Every build ends with an equivalence self-check: the four metric lists are
computed once from the processed directory and once from the freshly written results directory,
for every SR model id present. On any mismatch the results directory is deleted and the build
fails (`ResultsBuildError`). Slimming is therefore never a silent approximation.

`lr.json` (schema_version 1) replaces the per-config `lr_result.json` tree:

```json
{"schema_version": 1,
 "configs": {"M1R1-M1R1-M1R1": {
   "top5": ["…"], "top5_indices": [1, 14, 12, 2, 8],
   "responses": [{"model": "gpt-5-nano-2025-08-07",
                  "input_tokens": 7435, "cached_tokens": 1152, "output_tokens": 474}],
   "usage_totals": {"input_tokens": 7435, "cached_tokens": 1152,
                    "output_tokens": 474, "cost_usd": 0.000504}}}}
```

`meta.json` records `benchmark`, `project`, `bug_id`, `source`, and the two processed-tree signals
the BugsInPy exclusions report needs (`has_sr_result`, `trigger_blank`). It deliberately contains
no timestamps: output is written with sorted keys so an unchanged processed tree rebuilds
byte-for-byte, and `git diff` stays empty.

A rebuild reflects the processed tree **as it is on disk**: `--force` deletes the bug's results
directory first, so rebuilding from a partially extracted bug (no Agent4SR run, say) yields a
correspondingly reduced results directory, and drops the per-bug `evaluation/` CSVs until
`run_evaluation.py` runs again. Rebuild from complete processed data, or not at all.

Full LLM records are copied out of band into the gitignored
`transcripts/<BM>/<Project>/<BugId>/{Agent4LR/<config>,Agent4SR/<model>}/` tree.

Benchmark-level snapshots live in `results/<BM>/_meta/`. For BugsInPy that is `audit.csv` (the
extraction audit, which drives exclusion classification) and `bugsinpy-index.csv` (so project and
bug enumeration works without the benchmark clone or a running container).

---

## 8. Stage 5 — Evaluation

CLI: `fl_methods/run_evaluation.py`. Implementation: `src/evaluation/`. **Reads `results/` only** —
it never touches `data/`.

### Metrics

Defined in `src/evaluation/metrics.py`:

- **FR** — lowest 1-based rank of any faulty entity.
- **AR** — mean rank over all faulty entities.
- **Top-K**, K ∈ {1…5} — 1 iff FR ≤ K.
- **WE** — `(rank_of_last_fault − n_faulty) / n_healthy`: the fraction of healthy methods one must
  inspect to catch every fault.

The candidate **universe** is the SR top-20 for the selected `sr_model_id`; a bug whose top-20 is
missing has no universe and is skipped. The `*_first` variants evaluate against `faults_first.csv`.

### Per-bug output

Written under `results/<BM>/<Project>/<BugId>/evaluation/`:

| File | Rows | Sample |
|---|---|---|
| `baselines.csv` | Ochiai, SBIR, BoostN against `faults.csv` | `Ochiai,1.0,1.0,1,1,1,1,1,0.0,0,0,0,` |
| `baselines_first.csv` | the same, against `faults_first.csv` | — |
| `flexfl.csv` | every Agent4LR config against `faults.csv` | `Agent4LR-M1R1-M1R1-M1R1,1.0,1.0,1,1,1,1,1,0.0,7435,1152,474,0.000504` |
| `flexfl_first.csv` | the same, against `faults_first.csv` | — |

Schema: `Method,FR,AR,Top1,Top2,Top3,Top4,Top5,WE,InputTokens,CachedTokens,OutputTokens,CostUSD`.
Token and cost columns are populated for `flexfl*` rows only.

### Cross-bug output

Written under `evaluation/<BM>/`:

| File | Granularity | Schema |
|---|---|---|
| `baselines.csv`, `baselines_first.csv` | one row per (bug, method) | `Project,BugId,` + the per-bug columns |
| `flexfl.csv`, `flexfl_first.csv` | one row per (bug, config) | the same |
| `<slot>_summary.csv` | one row per method/config, averaged over bugs | `Method,Bugs,MFR,MAR,Top{1..5}Rate,MeanWE,TotalInputTokens,TotalCachedTokens,TotalOutputTokens,TotalCostUSD,MeanCostUSD` |
| `exclusions.csv` (BugsInPy) | one row per excluded bug | `Project,BugId,Bucket,Scope,…` |

Re-running a bug replaces its rows **in place** in the long CSVs, so row order is stable and the
committed files change only where the numbers do.

Exclusion buckets (`src/evaluation/exclusions.py`) classify why a BugsInPy bug is not fully
evaluable — extraction incomplete, FL unreachable, method unlocalizable, SR not run, LR readiness
or measurement skipped, LR not run — with the extraction audit taking precedence over on-disk
state.

---

## 9. Identifier formats

These strings are load-bearing; every consumer keys off them.

| Kind | Format | Example |
|---|---|---|
| Statement ID | `pkg.Class#line` | `org.apache.commons.lang3.time.FastDatePrinter#262` |
| Method ID (corpus) | `pkg$Class.method(SimpleParams)` | `org.apache.commons.lang3.math$NumberUtils.createNumber(String)` |
| Method ID (dotted) | `pkg.Class.method(SimpleParams)` | used in `top20/*.txt` and LLM output; canonicalized on read |
| Method ID (FlexFL export) | `pkg.Class.method(Params).startLine.endLine` | `run_flexfl.py` compatibility exports |
| Test ID | `Class#method` *and* `Class::method` | both are kept; consumers dispatch on the form |

A new benchmark should map its natural naming into these shapes rather than introduce a new one.
For Python, `src/common/python_parser.py` builds `<module>$<QualName>.<func>(<params>)`, which
keeps the same package/class boundary semantics.

---

## 10. Adapter API

Registering a benchmark takes three pieces.

**The adapter** — mirror `src/benchmarks/defects4j/adapter.py`:

```python
@dataclass(frozen=True)
class XAdapter:
    benchmark_key: str  # CLI key, lowercase, e.g. "bugsinpy"
    benchmark_folder: str  # on-disk folder under data/, e.g. "BugsInPy"

    def list_projects(self) -> list[str]: ...
    def list_cases(self, project: str) -> list[int]: ...
    def build_repo(self, project: str, case_id: int, *, buggy: bool = True) -> XRepo: ...
```

Register it in `src/benchmarks/registry.py`.

**The repo type** — duck-typed today, derived from how `D4JRepo` is consumed in
`run_extraction.py` and `src/extraction/pipeline.py`. It must expose the attributes `project`,
`bug_id`, `buggy`, `version_flag`, `repo_dir`, `output_dir`, and the methods `is_checked_out`,
`checkout`, `is_compiled`, `compile`, `export_property`, `export_all_properties`,
`get_src_class_dir`, `get_bin_class_dir`, `get_src_tests_dir`, `get_bin_tests_dir`,
`get_cp_test`, `get_modified_classes`, `get_relevant_classes`, `get_relevant_test_classes`,
`get_trigger_tests`, `classpath_from_class_signature`, `get_all_test_methods`,
`get_relevant_test_methods`, `remove_repo`. Many of the accessors are thin readers over the
property files in `output_dir` and can be shared.

**The layout alias** — add an entry to `src/core/layout.py:BENCHMARK_ALIASES`:

```python
BENCHMARK_ALIASES: dict[str, str] = {
    "d4j": "D4J",
    "defects4j": "D4J",
    "bugsinpy": "BugsInPy",
    "bip": "BugsInPy",
}
```

Path resolution (`DatasetLayout.processed_root(benchmark)`, the `get_*` helpers in
`src/common/config.py`) and the `--benchmark <key>` flag then work uniformly.

---

## 11. Generalization map

What is Java/Defects4J-specific today, and what replaces it:

| Concern | Today | Replacement |
|---|---|---|
| Coverage tool | GZoltar → `sfl/sfl/txt/*` | FauxPy → `FauxPy/{coverage,reports}/` (already implemented for Python); any tool emitting the §4.4 schemas. |
| Source parser | `javalang` via `src/common/java_parser.py` | `ast` via `src/common/python_parser.py`; `tree-sitter-<lang>` in general. Must produce the same `MethodInfo`/`StatementInfo` shapes. |
| Build/test driver | `defects4j compile` / `test` | Any project-native runner invoked from the adapter's `Repo`. |
| Property exports | `defects4j export -p <prop>` | Read the benchmark's own metadata; resolve locations through `config.py`. |
| Diff-based fault extraction | Unix `diff` in `faults.py` | Already language-agnostic once `method_signatures.csv` is correct. |
| Bug-report sources | Host-dispatched parser in `report_parser.py` | Extend the dispatcher, or ship cached JSON. |
| Stopwords | `src/boostn/StopwordsPlusJava.txt` | A language-appropriate list wired in through `text_processor.load_stopwords`. |
| Container | `defects4j:cefl` / `bugsinpy:cefl` images | Mirror `utils/docker/<benchmark>/`; names come from constants and env overrides, never hardcoded call sites. |

**Bottom line:** the schemas in §4–§8 are the contract, the tools that produce them are
interchangeable, and tool-native *filenames* are resolved through `config.py` rather than assumed.

---

## 12. Checklist

Per bug, under `data/<BM>/processed/<Project>/<BugId>/`:

- [ ] Metadata (§4.1): source/binary directory paths, test classpath, modified and relevant
      classes, relevant and triggering tests — as native values, however named.
- [ ] Tests (§4.2): `all_tests.txt`, `relevant_tests.txt`, the coverage tool's test list,
      failing/trigger dumps, and ideally `trigger_test_clean.txt`.
- [ ] `method_signatures.csv` (§4.3) — `corpus_id;path;startLine;endLine`. Load-bearing.
- [ ] Coverage (§4.4): `spectra.csv`, `matrix.txt`, `tests.csv`, `ochiai.ranking.csv`.
- [ ] Faults (§4.5): `faults.csv` + `faults.txt`, `faults_first.csv` + `faults_first.txt`.
- [ ] `bug_report.json` (§4.6) with `url`, `title`, `description`, `raw`.

Then the rest runs unchanged: `FL/{Ochiai,BoostN,SBIR}/`, `FlexFL/SR/` (corpus, Agent4SR records,
rankings), `FlexFL/LR/Agent4LR/<config>/`, `results/<BM>/…` via `run_build_results.py`, and
`evaluation/<BM>/` via `run_evaluation.py`.

Python wiring: an adapter under `src/benchmarks/<name>/`, its registry entry, a
`BENCHMARK_ALIASES` alias, `utils/docker/<name>/` scripts, and a language-appropriate stopwords
file if BoostN or Blues will run.
