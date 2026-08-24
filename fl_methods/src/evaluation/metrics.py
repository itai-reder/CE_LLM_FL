"""Per-bug evaluation metrics: FR, AR, Top-K, Wasted Effort.

All metrics consume a rank map (``dict[K, float]``) covering a candidate
universe, plus the set of faulty keys. ``None`` is returned when a metric is
inapplicable (no faulty key in universe, denominator zero, etc.) and the
writer renders that as a blank CSV cell.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Set
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def first_rank(ranks: Mapping[K, float], faulty: Set[K]) -> float | None:
    """Lowest rank among faulty keys present in ``ranks``.

    Returns ``None`` when no faulty key has a rank (e.g. universe was filtered
    to exclude every faulty member).
    """
    hits = [ranks[k] for k in faulty if k in ranks]
    return min(hits) if hits else None


def mean_rank(ranks: Mapping[K, float], faulty: Set[K]) -> float | None:
    """Arithmetic mean rank over faulty keys present in ``ranks``."""
    hits = [ranks[k] for k in faulty if k in ranks]
    return (sum(hits) / len(hits)) if hits else None


def top_k(rank: float | None, k: int) -> int:
    """Return ``1`` when ``rank`` exists and is at most ``k``, else ``0``.

    Uses ``<=`` so that a tied group sitting exactly on the boundary (e.g. an
    average rank of ``5.0`` for k=5) counts as a hit.
    """
    if rank is None:
        return 0
    return 1 if rank <= k else 0


def wasted_effort(
    ranks: Mapping[K, float],
    faulty: Set[K],
    universe_size: int,
) -> float | None:
    """Fraction of healthy candidates inspected before finding all faults.

    Formula (per user spec): ``(rank_last_fault - |F|) / n_healthy``, where
    ``n_healthy = universe_size - |F|`` and ``rank_last_fault`` is the
    maximum rank among faulty hits. Returns ``None`` when no faulty key has a
    rank or when ``n_healthy <= 0`` (denominator would vanish — the universe
    is entirely faulty).

    Under tied-rank semantics the numerator can be negative (e.g. all faults
    share rank 1 with |F|=2 -> 1 - 2 = -1). That's a faithful signal that
    faults cluster at the top; callers may clamp to 0 for display.
    """
    hits = sorted(ranks[k] for k in faulty if k in ranks)
    if not hits:
        return None
    n_faulty = len(hits)
    n_healthy = universe_size - n_faulty
    if n_healthy <= 0:
        return None
    rank_last = hits[-1]
    return (rank_last - n_faulty) / n_healthy
