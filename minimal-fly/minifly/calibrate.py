"""Single-neuron calibration.

These look at one neuron at a time, never at behaviour: the tonic MN drive
that gives 40 Hz, and the ORN gain that gives 50 Hz at concentration K.
"""
import numpy as np

from . import config as cfg
from .neuron import Network


def mean_rate(drive: float, sigma: float, key: int, n_copies: int = 100,
              duration_ms: float = 5000.0, warmup_ms: float = 200.0) -> float:
    """Mean firing rate (Hz) of an unconnected neuron under constant drive
    plus noise. The same noise draws are used for every drive value."""
    net = Network(np.zeros((1, 1, 1)), [[cfg.LIF["v_th"]]], [sigma],
                  np.zeros(n_copies, dtype=np.int64))
    rng = np.random.default_rng(np.random.SeedSequence(
        [cfg.SEED_BASE, cfg.STREAM_CALIBRATION, 50_000 + key]))
    d = np.full((n_copies, 1), float(drive))
    steps = int(round(duration_ms / cfg.DT_MS))
    warm = int(round(warmup_ms / cfg.DT_MS))
    count = 0
    for s in range(steps):
        fired = net.step(s, d, rng.standard_normal((n_copies, 1)))
        if s >= warm:
            count += int(fired.sum())
    return count / (n_copies * (steps - warm) * cfg.DT_MS / 1000.0)


def drive_for_rate(target_hz: float, sigma: float, key: int,
                   lo: float = 0.0, hi: float = 20.0, tol: float = 1e-4):
    """Bisection for the constant drive (mV/ms) giving target_hz."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mean_rate(mid, sigma, key) < target_hz:
            lo = mid
        else:
            hi = mid
    drive = 0.5 * (lo + hi)
    return drive, mean_rate(drive, sigma, key)
