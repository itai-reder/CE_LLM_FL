"""Analysis frames for the swap-expanded grid.

Reads the committed cross-bug evaluation tables (``evaluation/<BM>/flexfl.csv``),
restricts them to the 39 swap-expanded configurations (see
``analysis/swap-expanded.md``) and to the bug panel where every configuration
produced a non-blank First Rank, and returns two frames:

* ``per_bug`` — one row per (bug, config), annotated with the configuration's
  role decomposition (planner / tool-caller / finisher), its fixed background
  tier, the swapped role and the swept variant.
* ``summary_by_config`` — one row per configuration: mean metrics (MFR, MAR,
  Top-N rates, WE), token and cost totals, and the blended price rate.

Nothing is written to disk; the notebook builds what it needs in memory.

    PYTHONPATH=fl_methods python analysis/frames.py --benchmark bip
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_DIRS = {"d4j": "D4J", "bip": "BIP", "D4J": "D4J", "BIP": "BIP"}

ROLES = ("planner", "tool_caller", "finisher")


def eval_csv(benchmark: str = "d4j", name: str = "flexfl") -> Path:
    """Path to a committed cross-bug evaluation table."""
    return REPO_ROOT / "evaluation" / BENCHMARK_DIRS[benchmark] / f"{name}.csv"


# ---------------------------------------------------------------------------
# The 39-config cut grid (mirrors _swap_expanded_configs() in
# fl_methods/run_experiment.py — 3 fixed tiers x 3 roles x 5 swept variants,
# with the homogeneous anchors shared across roles)
# ---------------------------------------------------------------------------


def swap_expanded_configs() -> list[str]:
    """Canonical ``Planner-ToolCaller-Finisher`` strings in sweep order."""
    configs: list[str] = []
    for i in (1, 2, 3):
        anchor = f"M{i}R1"
        for m in (1, 2, 3):
            for r in (1, 2, 3):
                # Cut: only M2 sweeps R1..R3; M1/M3 stay pinned at R1.
                if m != 2 and r != 1:
                    continue
                swap = f"M{m}R{r}"
                configs.append(f"{swap}-{anchor}-{anchor}")  # planner
                configs.append(f"{anchor}-{swap}-{anchor}")  # tool_caller
                configs.append(f"{anchor}-{anchor}-{swap}")  # finisher
    return list(dict.fromkeys(configs))


ALL_CONFIGS = swap_expanded_configs()
N_CONFIGS = len(ALL_CONFIGS)
assert N_CONFIGS == 39, f"expected 39 swap-expanded configs, got {N_CONFIGS}"

LABEL_OF = {c: f"Agent4LR-{c}" for c in ALL_CONFIGS}


def classify(config: str) -> tuple[str, int, str]:
    """Return ``(swapped_role, fixed_tier, swept_variant)`` for a config.

    Homogeneous anchors ``M<i>R1`` x3 classify as ``("anchor", i, "M<i>R1")``.
    Otherwise exactly one role differs from the ``M<i>R1`` background pair —
    that role is the swapped one and its value is the swept variant.
    """
    slots = config.split("-")
    if len(slots) != 3:
        raise ValueError(f"malformed config: {config!r}")
    if slots[0] == slots[1] == slots[2]:
        return "anchor", int(slots[0][1]), slots[0]
    for role, idx in zip(ROLES, range(3), strict=True):
        others = [slots[j] for j in range(3) if j != idx]
        if others[0] == others[1] and slots[idx] != others[0]:
            return role, int(others[0][1]), slots[idx]
    raise ValueError(f"config does not fit the single-swap grid: {config!r}")


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def load_swap_expanded(benchmark: str = "d4j") -> pd.DataFrame:
    """Long evaluation rows restricted to the 39 swap-expanded configs."""
    df = pd.read_csv(eval_csv(benchmark))
    return df[df["Method"].isin(set(LABEL_OF.values()))].copy()


def full_sweep_bug_set(df: pd.DataFrame) -> set[tuple[str, object]]:
    """Bugs where every one of the 39 configs has a non-blank FR."""
    expected = set(LABEL_OF.values())
    fr_ok = df[df["FR"].notna()]
    seen = fr_ok.groupby(["Project", "BugId"])["Method"].apply(set)
    return {bug for bug, methods in seen.items() if expected.issubset(methods)}


def restrict_to_bugs(df: pd.DataFrame, bugs: set[tuple[str, object]]) -> pd.DataFrame:
    mask = [(p, b) in bugs for p, b in zip(df["Project"], df["BugId"], strict=True)]
    return df[mask].copy()


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add config-decomposition columns to a long evaluation frame."""
    df = df.copy()
    config = df["Method"].str.replace("Agent4LR-", "", regex=False)
    slots = config.str.split("-", expand=True)
    df["Config"] = config
    df["Planner"] = slots[0]
    df["ToolCaller"] = slots[1]
    df["Finisher"] = slots[2]
    classified = config.map({c: classify(c) for c in ALL_CONFIGS})
    df["SwappedRole"] = [c[0] for c in classified]
    df["FixedTier"] = [c[1] for c in classified]
    df["SweptVariant"] = [c[2] for c in classified]
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS = [
    "Method",
    "Config",
    "Planner",
    "ToolCaller",
    "Finisher",
    "SwappedRole",
    "FixedTier",
    "SweptVariant",
    "Bugs",
    "MFR",
    "MAR",
    "Top1Rate",
    "Top2Rate",
    "Top3Rate",
    "Top4Rate",
    "Top5Rate",
    "MeanWE",
    "TotalInputTokens",
    "TotalCachedTokens",
    "TotalOutputTokens",
    "TotalCostUSD",
    "MeanCostUSD",
    "MeanPricePerMTokens",
]


def _price_per_m(cost_total: float, input_tokens: int, output_tokens: int) -> float:
    """Effective $/1M tokens: total cost over total (input + output) tokens.

    Cached tokens are a subset of input (matching OpenAI billing), so they do
    not enter the denominator. Returns 0.0 when no tokens were spent.
    """
    total_tokens = int(input_tokens) + int(output_tokens)
    if total_tokens <= 0:
        return 0.0
    return cost_total / total_tokens * 1_000_000


def build_summary(df_long: pd.DataFrame, order: list[str] = ALL_CONFIGS) -> pd.DataFrame:
    """One row per config: mean metrics + token/cost totals over its bugs."""
    rows: list[dict] = []
    for method, sub in df_long.groupby("Method"):
        fr_present = sub[sub["FR"].notna()]
        bugs = len(fr_present)
        rows.append(
            {
                "Method": method,
                "Config": sub["Config"].iloc[0],
                "Planner": sub["Planner"].iloc[0],
                "ToolCaller": sub["ToolCaller"].iloc[0],
                "Finisher": sub["Finisher"].iloc[0],
                "SwappedRole": sub["SwappedRole"].iloc[0],
                "FixedTier": sub["FixedTier"].iloc[0],
                "SweptVariant": sub["SweptVariant"].iloc[0],
                "Bugs": bugs,
                "MFR": fr_present["FR"].mean() if bugs else math.nan,
                "MAR": fr_present["AR"].mean() if bugs else math.nan,
                "Top1Rate": fr_present["Top1"].mean() if bugs else math.nan,
                "Top2Rate": fr_present["Top2"].mean() if bugs else math.nan,
                "Top3Rate": fr_present["Top3"].mean() if bugs else math.nan,
                "Top4Rate": fr_present["Top4"].mean() if bugs else math.nan,
                "Top5Rate": fr_present["Top5"].mean() if bugs else math.nan,
                "MeanWE": fr_present["WE"].mean() if bugs else math.nan,
                "TotalInputTokens": int(fr_present["InputTokens"].sum()),
                "TotalCachedTokens": int(fr_present["CachedTokens"].sum()),
                "TotalOutputTokens": int(fr_present["OutputTokens"].sum()),
                "TotalCostUSD": float(fr_present["CostUSD"].sum()),
                "MeanCostUSD": (float(fr_present["CostUSD"].mean()) if bugs else math.nan),
                "MeanPricePerMTokens": _price_per_m(
                    float(fr_present["CostUSD"].sum()),
                    int(fr_present["InputTokens"].sum()),
                    int(fr_present["OutputTokens"].sum()),
                ),
            }
        )
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    cat = pd.CategoricalIndex(df["Config"], categories=order, ordered=True)
    df = df.assign(_order=cat).sort_values("_order").drop(columns="_order")
    return df.reset_index(drop=True)


def build_frames(benchmark: str = "d4j") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(per_bug_long, summary_by_config)`` for the 39-config sweep."""
    df = load_swap_expanded(benchmark)
    bugs = full_sweep_bug_set(df)
    df = annotate(restrict_to_bugs(df, bugs))
    return df, build_summary(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the swap-expanded panel sizes")
    parser.add_argument("--benchmark", choices=("d4j", "bip"), default="d4j")
    args = parser.parse_args()

    per_bug, summary = build_frames(args.benchmark)
    n_bugs = per_bug[["Project", "BugId"]].drop_duplicates().shape[0]
    print(f"{N_CONFIGS} configs x {n_bugs} full-sweep bugs -> {len(per_bug)} rows")

    mismatched = summary[summary["Bugs"] != n_bugs]
    if not mismatched.empty:
        raise RuntimeError(f"summary has non-{n_bugs} bug counts:\n{mismatched}")
    assert len(summary) == N_CONFIGS, f"expected {N_CONFIGS} configs, got {len(summary)}"


if __name__ == "__main__":
    main()
