"""The 8-neuron chemotaxis circuit and its perturbations.

An edge is (pre, post, sign);
every edge carries the same synapse count n, so its weight is
sign * n * w_syn mV per presynaptic spike.
"""
import numpy as np

from . import config as cfg

# Every edge from a given neuron has that neuron's sign.
SIGN_BY_PRE = {cfg.ORN_L: 1, cfg.ORN_R: 1, cfg.PN_L: 1, cfg.PN_R: 1,
               cfg.IN_L: -1, cfg.IN_R: -1}

BASELINE = {
    "e1": (cfg.ORN_L, cfg.PN_L, +1),
    "e2": (cfg.ORN_R, cfg.PN_R, +1),
    "e3": (cfg.ORN_L, cfg.IN_L, +1),
    "e4": (cfg.ORN_R, cfg.IN_R, +1),
    "e5": (cfg.IN_L, cfg.PN_R, -1),
    "e6": (cfg.IN_R, cfg.PN_L, -1),
    "e7": (cfg.PN_L, cfg.MN_R, +1),
    "e8": (cfg.PN_R, cfg.MN_L, +1),
}
COMPARISON_EDGES = ("e5", "e6")      # contralateral inhibition inside the brain
CROSSED_MOTOR_EDGES = ("e7", "e8")   # crossed sensory-to-motor pathway


def edges_of(named: dict) -> tuple:
    return tuple(named.values())


def weight_matrix(edges, n_syn: float) -> np.ndarray:
    """(8, 8) [pre, post] matrix in mV per presynaptic spike."""
    w = np.zeros((cfg.N_NEURONS, cfg.N_NEURONS))
    for pre, post, sign in edges:
        w[pre, post] += sign * n_syn * cfg.LIF["w_syn"]
    return w


def flip_signs(named: dict, names) -> dict:
    """P1: the listed edges change sign."""
    return {k: (pre, post, -sign) if k in names else (pre, post, sign)
            for k, (pre, post, sign) in named.items()}


def drop(named: dict, names) -> dict:
    """P3a / P3b: the listed edges are removed."""
    return {k: e for k, e in named.items() if k not in names}


def thresholds(dose_mv: float = 0.0, neurons=None) -> np.ndarray:
    """(8,) thresholds, raised by dose_mv on the listed neurons (all if None)."""
    v_th = np.full(cfg.N_NEURONS, cfg.LIF["v_th"])
    idx = range(cfg.N_NEURONS) if neurons is None else neurons
    for i in idx:
        v_th[i] += dose_mv
    return v_th


SENSORY_AND_INTER = (cfg.ORN_L, cfg.ORN_R, cfg.PN_L, cfg.PN_R, cfg.IN_L, cfg.IN_R)
