"""Looming stimulus and the LC4/LPLC2 feature encoding.

This module is the ONLY part of the model not derived from connectome data.
Following Ache et al. 2019: LC4 encodes angular velocity (linear), LPLC2
encodes angular size (Gaussian, peaked near 42 deg). The transformation from
these two signals to the giant fiber is NOT written here -- it is left to the
measured connectome weights.
"""
import numpy as np

from . import config
from .circuit import Circuit

LPLC2_PEAK_DEG = 42.0


def angular_size_deg(t_ms: np.ndarray, rv_ms: float) -> np.ndarray:
    """theta(t) = 2*arctan(r / (v*|t|)), with t < 0 before collision."""
    return np.degrees(2.0 * np.arctan(rv_ms / np.abs(t_ms)))


def _time_of_theta(theta_deg: float, rv_ms: float) -> float:
    """Invert angular_size_deg; returns a negative time (before collision)."""
    return -rv_ms / np.tan(np.radians(theta_deg) / 2.0)


def time_grid(rv_ms: float, dt_ms: float) -> np.ndarray:
    start = _time_of_theta(config.THETA_START_DEG, rv_ms)
    stop = _time_of_theta(config.THETA_END_DEG, rv_ms)
    return np.arange(start, stop, dt_ms)


def _reference_peak_velocity() -> float:
    """Peak angular velocity of the fastest condition, in deg/ms.

    LC4 drive is normalised against this single fixed value rather than
    per condition. Normalising within each r/v would pin LC4's peak to 1.0
    at every approach speed and erase the 8x range that benchmark B2 tests.
    """
    rv = min(config.RV_VALUES_MS)
    t = time_grid(rv, config.DT_MS)
    return float(np.gradient(angular_size_deg(t, rv), config.DT_MS).max())


_LC4_REFERENCE_PEAK_VELOCITY_DEG_MS = _reference_peak_velocity()


def input_drive(c: Circuit, rv_ms: float, dt_ms: float,
                sigma_deg: float, eye: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron drive in mV/ms. Only LC4 and LPLC2 receive input.

    A looming stimulus on the left of the fly drives the LEFT-side visual
    neurons; 'both' drives all of them.
    """
    if eye not in {"left", "right", "both"}:
        raise ValueError(f"eye must be left/right/both, got {eye!r}")

    t = time_grid(rv_ms, dt_ms)
    theta = angular_size_deg(t, rv_ms)
    d_theta = np.gradient(theta, dt_ms)          # deg/ms

    velocity_signal = d_theta / _LC4_REFERENCE_PEAK_VELOCITY_DEG_MS    # LC4, linear in angular velocity
    size_signal = np.exp(                        # LPLC2, Gaussian in angular size
        -0.5 * ((theta - LPLC2_PEAK_DEG) / sigma_deg) ** 2
    )

    drive = np.zeros((c.n_neurons, t.size), dtype=np.float64)
    side_ok = np.ones(c.n_neurons, dtype=bool)
    if eye != "both":
        side_ok = c.sides == ("L" if eye == "left" else "R")

    drive[c.mask_by_type("LC4") & side_ok] = velocity_signal
    drive[c.mask_by_type("LPLC2") & side_ok] = size_signal
    return drive, t


def apply_input_jitter(drive: np.ndarray, c: Circuit,
                       rng: np.random.Generator) -> np.ndarray:
    """Desynchronise the visual input layer by per-neuron temporal offsets.

    Returns a new array; `drive` is not modified. Each LC4/LPLC2 row is
    shifted by an integer number of steps drawn uniformly from
    +/- INPUT_JITTER_MS, with zero padding at the edge the row rolls off.
    Non-visual rows are left untouched.
    """
    jittered = drive.copy()
    n = int(config.INPUT_JITTER_MS / config.DT_MS)
    for i in np.flatnonzero(c.mask_by_type("LC4", "LPLC2")):
        offset = int(rng.integers(-n, n + 1))
        jittered[i] = np.roll(jittered[i], offset)
        if offset > 0:
            jittered[i, :offset] = 0.0
        elif offset < 0:
            jittered[i, offset:] = 0.0
    return jittered
