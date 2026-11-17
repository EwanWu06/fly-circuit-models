"""Second version: 20 neurons with random wiring.

Neurons 0-6 smell at the left antenna, 7-13 at the right, 14-17 are
interneurons, 18 and 19 drive the left and right legs. Wiring is drawn at
random in a fixed order; synaptic strengths reuse the v1 frozen
synapse count without any new tuning.
"""
import numpy as np

from . import config as cfg
from .sim import Layout

N = 20
ORN_LEFT = tuple(range(0, 7))
ORN_RIGHT = tuple(range(7, 14))
ORNS = ORN_LEFT + ORN_RIGHT
INTS = (14, 15, 16, 17)
MN_L, MN_R = 18, 19
MNS = (MN_L, MN_R)
LAYOUT = Layout(N, ORN_LEFT, ORN_RIGHT, MN_L, MN_R)
NAMES = ([f"ORN_L{i + 1}" for i in range(7)] + [f"ORN_R{i + 1}" for i in range(7)]
         + ["INT1", "INT2", "INT3", "INT4", "MN_L", "MN_R"])

P_EDGE = 0.5
ORNS_PER_SIDE = 7
STREAM_TRIALS = 3        # starts and noise for every v2 trial
STREAM_WIRING = 4        # random wiring
N_CIRCUITS = 1000
N_TRIALS = 100
N_SENSITIVITY = 300
SEPS_MM = (0.3, 0.6, 1.25, 2.5, 5.0)
N_SEP_TRIALS = 200


def random_circuit(k: int):
    """Circuit k: (edges, interneuron signs). Draw order, as registered:
    the four signs, then ORN -> INT, then INT -> INT, then INT -> MN."""
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.SEED_BASE, STREAM_WIRING, k]))
    signs = np.where(rng.random(len(INTS)) < 0.5, 1, -1)
    edges = []
    for i in ORNS:
        for j in INTS:
            if rng.random() < P_EDGE:
                edges.append((i, j, 1))
    for a, j in enumerate(INTS):
        for j2 in INTS:
            if j2 != j and rng.random() < P_EDGE:
                edges.append((j, j2, int(signs[a])))
    for a, j in enumerate(INTS):
        for m in MNS:
            if rng.random() < P_EDGE:
                edges.append((j, m, int(signs[a])))
    return edges, signs


def designed_circuit():
    """The v1 circuit on the v2 layout (positive control):
    INT1 = PN_L, INT2 = PN_R, INT3 = IN_L, INT4 = IN_R."""
    pn_l, pn_r, in_l, in_r = INTS
    edges = [(i, pn_l, 1) for i in ORN_LEFT] + [(i, in_l, 1) for i in ORN_LEFT]
    edges += [(i, pn_r, 1) for i in ORN_RIGHT] + [(i, in_r, 1) for i in ORN_RIGHT]
    edges += [(in_l, pn_r, -1), (in_r, pn_l, -1), (pn_l, MN_R, 1), (pn_r, MN_L, 1)]
    return edges, np.array([1, 1, -1, -1])


def weight_matrix(edges, n_syn: float, int_scale: float = 1.0) -> np.ndarray:
    """ORN edges carry n_syn / 7 synapses (seven together equal one v1 ORN);
    interneuron edges carry n_syn * int_scale."""
    w = np.zeros((N, N))
    for pre, post, sign in edges:
        n = n_syn / ORNS_PER_SIDE if pre in ORNS else n_syn * int_scale
        w[pre, post] += sign * n * cfg.LIF["w_syn"]
    return w


def thresholds() -> np.ndarray:
    return np.full(N, cfg.LIF["v_th"])


def crossed_index(edges, signs) -> int:
    """X = sum over interneurons of sign * (left inputs - right inputs) *
    (edge to MN_R - edge to MN_L); interneuron-to-interneuron edges ignored.
    Positive X: more odour on the left speeds the right legs, a left turn."""
    x = 0
    for a, j in enumerate(INTS):
        n_left = sum(1 for p, q, _ in edges if q == j and p in ORN_LEFT)
        n_right = sum(1 for p, q, _ in edges if q == j and p in ORN_RIGHT)
        d = (int(any(p == j and q == MN_R for p, q, _ in edges))
             - int(any(p == j and q == MN_L for p, q, _ in edges)))
        x += int(signs[a]) * (n_left - n_right) * d
    return x


def describe(edges) -> str:
    ints = sorted({p for p, _, _ in edges if p in INTS} | {q for _, q, _ in edges if q in INTS})
    parts = []
    for j in ints:
        left = sum(1 for p, q, _ in edges if q == j and p in ORN_LEFT)
        right = sum(1 for p, q, _ in edges if q == j and p in ORN_RIGHT)
        sign = next((s for p, _, s in edges if p == j), 0)
        outs = [NAMES[q] for p, q, _ in edges if p == j]
        kind = "+" if sign > 0 else "-" if sign < 0 else "?"
        parts.append(f"{NAMES[j]}({kind}) L{left}/R{right} -> {','.join(outs) or 'none'}")
    return "; ".join(parts)
