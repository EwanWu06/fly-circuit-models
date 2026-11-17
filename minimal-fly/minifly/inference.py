"""Statistics. Written by hand, as in escape-circuit/src/stats.py,
so the environment needs nothing beyond numpy."""
from math import comb, erfc, sqrt

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def mcnemar_greater(a, b) -> float:
    """One-sided exact McNemar test that paired successes a exceed b.

    a, b: boolean arrays over the same trials. Only discordant pairs count:
    p = P(X >= n10) for X ~ Binomial(n10 + n01, 1/2).
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(n10, n + 1))
    return min(1.0, tail / 2 ** n)


def mcnemar_two_sided(a, b):
    """Two-sided exact McNemar test on paired successes a, b.
    Returns (p, n10, n01): n10 counts pairs where only a succeeded."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    return binomial_two_sided(n10, n10 + n01), n10, n01


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided test of k successes in n trials against p = 1/2."""
    if n == 0:
        return 1.0
    lo = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * lo / 2 ** n)


def _ranks(v) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_greater(x, y, n_perm: int = 10_000, seed: int = 0):
    """Spearman rho and a one-sided permutation p for rho > 0."""
    rx, ry = _ranks(x), _ranks(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(rx * ry[rng.permutation(ry.size)]) for _ in range(n_perm)])
    return rho, float((np.sum(null >= rho) + 1) / (n_perm + 1))


def cochran_armitage_decreasing(doses, successes, totals) -> float:
    """One-sided Cochran-Armitage test for success falling with dose."""
    d = np.asarray(doses, float)
    x = np.asarray(successes, float)
    n = np.asarray(totals, float)
    big_n = n.sum()
    p_bar = x.sum() / big_n
    if p_bar <= 0.0 or p_bar >= 1.0:
        return 1.0
    t = float(np.sum(d * (x - n * p_bar)))
    var = p_bar * (1 - p_bar) * (np.sum(n * d * d) - np.sum(n * d) ** 2 / big_n)
    z = t / sqrt(var)
    return 0.5 * erfc(-z / sqrt(2.0))      # P(Z <= z)


def holm(p_values, alpha: float = 0.05):
    """Holm step-down. Returns (adjusted p, rejected)."""
    p = np.asarray(p_values, float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj, adj <= alpha


def bh(p_values, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg, as in escape-circuit/src/stats.py. Returns rejections."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1)
    passed = p[order] <= alpha * ranks / p.size
    rejected = np.zeros(p.size, dtype=bool)
    if passed.any():
        rejected[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return rejected


def bootstrap_mean_ci(values, n_boot: int = 10_000, seed: int = 0,
                      level: float = 0.95):
    """Percentile bootstrap interval for the mean."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def z_test_two_sided(values) -> float:
    """Two-sided normal-approximation test that the mean is zero."""
    v = np.asarray(values, float)
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    if sd == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    z = mean / (sd / sqrt(v.size))
    return erfc(abs(z) / sqrt(2.0))
