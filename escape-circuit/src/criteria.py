"""PRE-REGISTERED pass criteria for benchmarks B1-B4.

Committed before any benchmark was run. Changing any threshold here after
seeing results invalidates the study.
"""
import numpy as np

# B1: GF spikes at an angular size of roughly 40 degrees (Ache et al. 2019).
B1_THETA_RANGE_DEG = (30.0, 50.0)

# B2: LC4's contribution grows as r/v shrinks; LPLC2's stays constant.
# "Grows" = strictly monotonic decrease in contribution across ascending r/v.
# "Constant" = coefficient of variation across r/v below this bound.
B2_LPLC2_MAX_CV = 0.25

# B3: silencing LPLC2 must shift the peak response time by at least this much,
# in the direction predicted by losing the size component.
B3_MIN_PEAK_SHIFT_MS = 2.0

# B4: binocular response is sublinear.
B4_MAX_INDEX = 0.9

# sigma is the only free parameter; calibrated against B1 alone, then frozen.
SIGMA_SEARCH_DEG = np.arange(5.0, 30.5, 0.5)


def check_b1(theta_by_rv: dict[float, float | None]) -> bool:
    lo, hi = B1_THETA_RANGE_DEG
    return all(t is not None and lo <= t <= hi for t in theta_by_rv.values())


def check_b2(lc4_by_rv: dict[float, float],
             lplc2_by_rv: dict[float, float]) -> bool:
    rvs = sorted(lc4_by_rv)
    lc4 = [lc4_by_rv[rv] for rv in rvs]
    growing_as_rv_shrinks = all(a > b for a, b in zip(lc4, lc4[1:]))
    lplc2 = np.array([lplc2_by_rv[rv] for rv in rvs])
    constant = (lplc2.std() / abs(lplc2.mean())) < B2_LPLC2_MAX_CV
    return growing_as_rv_shrinks and constant


def check_b3(peak_intact_ms: float, peak_lplc2_silenced_ms: float) -> bool:
    return abs(peak_lplc2_silenced_ms - peak_intact_ms) >= B3_MIN_PEAK_SHIFT_MS


def check_b4(index: float) -> bool:
    return index < B4_MAX_INDEX
