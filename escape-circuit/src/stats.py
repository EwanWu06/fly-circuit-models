"""Statistics for comparing the real circuit against rewired nulls."""
import numpy as np


def z_score(observed: float, null: np.ndarray) -> float:
    """Return the z-score of observed against the null distribution.

    For a constant null (all elements identical), returns 0.0 if observed
    equals that value, else returns infinity with the sign of observed - value.
    """
    # Detect constant null by comparing values themselves, not computed mean/std
    if null.size > 0 and np.all(null == null[0]):
        null_value = float(null[0])
        if observed == null_value:
            return 0.0
        else:
            return np.inf if observed > null_value else -np.inf

    # Standard z-score for non-constant null
    mean = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    if sd == 0.0:
        return 0.0 if observed == mean else float("inf")
    return (observed - mean) / sd


def empirical_p(observed: float, null: np.ndarray) -> float:
    """Two-tailed empirical p with the (r+1)/(n+1) correction."""
    centre = float(np.mean(null))
    n_extreme = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    return min(1.0, (n_extreme + 1) / (null.size + 1))


def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean array of rejections."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1)
    passed = p[order] <= alpha * ranks / p.size
    rejected = np.zeros(p.size, dtype=bool)
    if passed.any():
        rejected[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return rejected
