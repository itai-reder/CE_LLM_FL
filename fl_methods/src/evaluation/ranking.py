"""Score-to-rank primitives with average tie-breaking and uniform-fill.

These are pure functions: take a score map and (optionally) a universe of
candidates, return rank assignments. No I/O, no FL-method specifics — all of
that lives in :mod:`src.evaluation.sources`.

Ranking conventions
-------------------
* Higher score → lower (better) rank.
* Tied scores share the **mean** of their 1-based positions. Example: items
  at positions 2, 3, 4, 5 with identical scores all get rank ``3.5``.
* Candidates absent from the score map (but present in the universe) are
  appended with a single uniform rank equal to the mean of the remaining
  positions. Example: 5 candidates ranked out of 200 → unranked rank ``103``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any, TypeVar

K = TypeVar("K", bound=Hashable)


def assign_average_ranks(
    scored: Iterable[tuple[K, float]],
    *,
    tiebreak: Callable[[K], Any] = lambda _k: 0,
) -> dict[K, float]:
    """Rank items by score descending; tied scores share their mean position.

    Parameters
    ----------
    scored:
        Iterable of ``(key, score)`` pairs.
    tiebreak:
        Secondary deterministic ordering key for ties — only affects the
        *order* of identical-rank entries (which is invisible in the result
        dict but matters for stable test fixtures).

    Returns
    -------
    Mapping from key to 1-based average rank.
    """
    items = sorted(scored, key=lambda kv: (-kv[1], tiebreak(kv[0])))
    ranks: dict[K, float] = {}

    start = 1
    i = 0
    while i < len(items):
        j = i
        score = items[i][1]
        while j < len(items) and items[j][1] == score:
            j += 1
        run = j - i
        avg = (2 * start + run - 1) / 2
        for k in range(i, j):
            ranks[items[k][0]] = avg
        start += run
        i = j
    return ranks


def append_unranked_with_uniform_rank(
    ranked: dict[K, float],
    universe: Iterable[K],
) -> dict[K, float]:
    """Extend a rank map so every universe member has a rank.

    Members absent from ``ranked`` all receive the same rank: the mean of the
    positions ``len(ranked)+1 .. |universe|``. Returns a new dict; does not
    mutate the input.

    If the universe equals the ranked set (or is smaller), the input is
    returned as a shallow copy.
    """
    universe_set = set(universe)
    missing = universe_set - ranked.keys()
    out: dict[K, float] = dict(ranked)
    if not missing:
        return out

    n_universe = len(universe_set | ranked.keys())
    n_ranked = len(ranked)
    if n_ranked >= n_universe:
        return out

    avg = (n_ranked + 1 + n_universe) / 2
    for key in missing:
        out[key] = avg
    return out


def score_universe(
    scored: dict[K, float],
    universe: Iterable[K],
    *,
    tiebreak: Callable[[K], Any] = lambda _k: 0,
) -> dict[K, float]:
    """Rank ``scored`` (filtered to ``universe``) and uniform-fill the rest.

    Convenience composition of :func:`assign_average_ranks` and
    :func:`append_unranked_with_uniform_rank` — what every FL-method
    evaluation actually wants.
    """
    universe_set = set(universe)
    filtered = [(k, v) for k, v in scored.items() if k in universe_set]
    ranks = assign_average_ranks(filtered, tiebreak=tiebreak)
    return append_unranked_with_uniform_rank(ranks, universe_set)
