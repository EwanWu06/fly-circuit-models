"""All degree-preserving wirings of the circuit (P4).

A wiring keeps every edge's source neuron, sign and synapse count and only
changes its target, with no self-loops and no duplicate edges. So every
neuron keeps its in-degree and out-degree. escape-circuit samples such
wirings by double-edge swaps; this circuit has few enough that all are listed.
"""
from itertools import permutations

from . import circuit
from . import config as cfg


def _key(edges) -> tuple:
    return tuple(sorted((pre, post) for pre, post, _ in edges))


def enumerate_wirings(edges=None) -> list:
    """Every distinct wiring, sorted, as tuples of (pre, post, sign)."""
    edges = circuit.edges_of(circuit.BASELINE) if edges is None else edges
    pres = [pre for pre, _, _ in edges]
    posts = [post for _, post, _ in edges]
    keys = set()
    for perm in set(permutations(posts)):
        pairs = list(zip(pres, perm))
        if any(a == b for a, b in pairs) or len(set(pairs)) < len(pairs):
            continue
        keys.add(tuple(sorted(pairs)))
    return [tuple((a, b, circuit.SIGN_BY_PRE[a]) for a, b in key)
            for key in sorted(keys)]


def index_of(wirings: list, edges) -> int:
    key = _key(edges)
    for i, w in enumerate(wirings):
        if _key(w) == key:
            return i
    raise KeyError("wiring not in the list")


def mirror(edges) -> tuple:
    """Swap the inputs of the two MNs: the network upstream is unchanged and
    every left-right steering command is reversed."""
    swap = {cfg.MN_L: cfg.MN_R, cfg.MN_R: cfg.MN_L}
    return tuple(sorted((pre, swap.get(post, post), sign)
                        for pre, post, sign in edges))


def degrees(edges):
    out_deg = [0] * cfg.N_NEURONS
    in_deg = [0] * cfg.N_NEURONS
    for pre, post, _ in edges:
        out_deg[pre] += 1
        in_deg[post] += 1
    return tuple(out_deg), tuple(in_deg)


def describe(edges) -> str:
    names = cfg.NEURONS
    return "; ".join(f"{names[a]}{'->' if s > 0 else '-|'}{names[b]}"
                     for a, b, s in sorted(edges))
