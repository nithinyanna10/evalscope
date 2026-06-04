"""Shared utilities for evalscope_ext."""

from typing import List


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation coefficient between two equal-length sequences.

    Pure-Python, no scipy dependency.  Returns nan when len(x) < 2.
    """
    n = len(x)
    if n < 2:
        return float('nan')

    def _ranks(seq: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: seq[i])
        rank = [0.0] * n
        for r, i in enumerate(order, start=1):
            rank[i] = float(r)
        return rank

    rx = _ranks(x)
    ry = _ranks(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))
