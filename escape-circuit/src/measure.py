"""Measurements M1-M4 on the giant fiber.

M1-M3 and max_depolarisation_mv read the MEAN of the two giant fibers'
potentials and refer to the first spike of EITHER giant fiber. The per-GF
helpers at the end of this module give each giant fiber's own values.
"""
import numpy as np

from . import config, stimulus
from .circuit import Circuit
from .lif import SimResult

GF_TYPE = "DNp01"


def gf_indices(c: Circuit) -> np.ndarray:
    """Indices of the DNp01 giant fibers in `c` (one per side in the real circuit)."""
    return np.flatnonzero(c.mask_by_type(GF_TYPE))


def _first_spike_step(c: Circuit, r: SimResult) -> int | None:
    """First step at which EITHER giant fiber spikes; None if neither does.

    M1, M2, M3 and max_depolarisation_mv all use this one time point, so they
    refer to whichever giant fiber spikes first.
    """
    gf_spikes = r.spikes[gf_indices(c)]
    steps = np.flatnonzero(gf_spikes.any(axis=0))
    return int(steps[0]) if steps.size else None


def m1_spike_latency_ms(c: Circuit, r: SimResult) -> float | None:
    """Time from stimulus onset to the first GF spike; None if silent."""
    step = _first_spike_step(c, r)
    return None if step is None else float(r.t_ms[step] - r.t_ms[0])


def m2_angular_size_at_spike_deg(c: Circuit, r: SimResult,
                                 rv_ms: float) -> float | None:
    """Angular size theta at the first spike of either GF; None if neither spikes."""
    step = _first_spike_step(c, r)
    if step is None:
        return None
    return float(stimulus.angular_size_deg(r.t_ms[step], rv_ms))


def m3_peak_response_time_ms(c: Circuit, r: SimResult) -> float:
    """Time of peak SUBTHRESHOLD membrane potential, before any spike.

    The trace is the mean of the two giant fibers, read before the first
    spike of either GF (the whole trace if neither spikes), and the time is
    measured from stimulus onset. If that mean trace never rises above its
    first sample -- giant fibers that never leave rest, or are only
    hyperpolarised -- its maximum is the first sample and this returns 0.0 ms.
    That is how benchmark B3's coded check passes trivially with LPLC2
    silenced at r/v = 40 ms: the silenced "peak time" is 0.0 ms, hundreds of
    milliseconds from the intact one.
    """
    gf = gf_indices(c)
    if gf.size == 0:
        raise ValueError("Circuit contains no DNp01 giant fiber neurons")
    trace = r.v[gf].mean(axis=0)
    step = _first_spike_step(c, r)
    if step == 0:
        raise ValueError("First GF spike at step 0: check gain and stimulus parameters")
    window = trace if step is None else trace[:step]
    return float(r.t_ms[int(window.argmax())] - r.t_ms[0])


def max_depolarisation_mv(c: Circuit, r: SimResult) -> float:
    """Maximum depolarisation from rest, in mV, of the TWO-GF MEAN trace.

    The trace is the mean of the two giant fibers' potentials, and the window
    ends before the first spike of EITHER giant fiber (the whole trace if
    neither spikes). This is R in M4 and the "contribution" in benchmark B2.
    Because the window ends before a spike, the value is capped near
    v_th - v_0 = 7 mV. Once one giant fiber spikes, the value is about
    (the leading GF, a little below threshold, + the lagging GF, wherever it
    is at that moment) / 2: it measures how close together the two giant
    fibers reach threshold, not how large the response is. This deviates from
    the registered definition; gf_max_depolarisation_mv gives per-GF values.
    """
    gf = gf_indices(c)
    if gf.size == 0:
        raise ValueError("Circuit contains no DNp01 giant fiber neurons")
    step = _first_spike_step(c, r)
    if step == 0:
        raise ValueError("First GF spike at step 0: check gain and stimulus parameters")
    trace = r.v[gf].mean(axis=0)
    window = trace if step is None else trace[:step]
    return float(window.max() - config.LIF["v_0"])


def m4_binocular_index(r_both: SimResult, r_left: SimResult,
                       r_right: SimResult, c: Circuit) -> float:
    """R(both) / [R(left) + R(right)]. Below 1 means sublinear.

    R is max_depolarisation_mv: the two-GF mean, cut at the first spike of
    either giant fiber. See its docstring for what that measures. NaN if both
    monocular responses are zero.
    """
    denominator = (max_depolarisation_mv(c, r_left)
                   + max_depolarisation_mv(c, r_right))
    if denominator == 0.0:
        return float("nan")
    return max_depolarisation_mv(c, r_both) / denominator


def gf_spike_rate(c: Circuit, r: SimResult) -> float:
    """Fraction of GF neurons that spiked at least once."""
    gf = gf_indices(c)
    if gf.size == 0:
        raise ValueError("Circuit contains no DNp01 giant fiber neurons")
    return float(r.spikes[gf].any(axis=1).mean())


# ---------------------------------------------------------------------------
# Per-GF values, added post hoc. These read each
# giant fiber separately; nothing above calls them, and no registered
# benchmark is scored on them.
# ---------------------------------------------------------------------------

def _gf_by_side(c: Circuit) -> dict[str, int]:
    """Index of each giant fiber, keyed by soma side ('L', 'R'), sorted by side."""
    gf = gf_indices(c)
    if gf.size == 0:
        raise ValueError("Circuit contains no DNp01 giant fiber neurons")
    sides = [str(s) for s in c.sides[gf]]
    if len(set(sides)) != len(sides):
        raise ValueError(f"Expected one giant fiber per side, got sides {sides}")
    return {side: int(i) for side, i in sorted(zip(sides, gf.tolist()))}


def gf_first_spike_steps(c: Circuit, r: SimResult) -> dict[str, int | None]:
    """Each giant fiber's OWN first-spike step, keyed by side; None if it never spikes.

    _first_spike_step, by contrast, is the first spike of either giant fiber.
    """
    out: dict[str, int | None] = {}
    for side, i in _gf_by_side(c).items():
        steps = np.flatnonzero(r.spikes[i])
        out[side] = int(steps[0]) if steps.size else None
    return out


def gf_max_depolarisation_mv(c: Circuit, r: SimResult) -> dict[str, float]:
    """Each giant fiber's maximum depolarisation from rest in its OWN window, in mV.

    A giant fiber's own window is its trace before its own first spike, or its
    whole trace if it never spikes. Keyed by side ('L', 'R'). Unlike
    max_depolarisation_mv, nothing is averaged across the two giant fibers and
    neither is cut at the other's spike. For a giant fiber that spikes, the
    value is still capped near v_th - v_0 = 7 mV.
    """
    first = gf_first_spike_steps(c, r)
    out: dict[str, float] = {}
    for side, i in _gf_by_side(c).items():
        step = first[side]
        if step == 0:
            raise ValueError("First GF spike at step 0: check gain and stimulus parameters")
        window = r.v[i] if step is None else r.v[i, :step]
        out[side] = float(window.max() - config.LIF["v_0"])
    return out


def gf_max_depolarisation_shared_window_mv(c: Circuit,
                                           r: SimResult) -> dict[str, float]:
    """Each giant fiber's maximum depolarisation in the SHARED window, in mV.

    The shared window is the one max_depolarisation_mv uses: before the first
    spike of either giant fiber (the whole trace if neither spikes). Keyed by
    side. These are the two per-GF terms behind the coded measure, which is
    the maximum of their mean trace: it is at most the mean of these two
    values, and equals it when both traces peak at the same step.
    """
    step = _first_spike_step(c, r)
    if step == 0:
        raise ValueError("First GF spike at step 0: check gain and stimulus parameters")
    out: dict[str, float] = {}
    for side, i in _gf_by_side(c).items():
        window = r.v[i] if step is None else r.v[i, :step]
        out[side] = float(window.max() - config.LIF["v_0"])
    return out
