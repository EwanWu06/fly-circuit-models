"""Perturbations used as controls. Only A is implemented this round."""
from dataclasses import replace
from typing import Protocol

import numpy as np

from .circuit import Circuit


class Perturbation(Protocol):
    name: str

    def apply(self, c: Circuit, fraction: float,
              rng: np.random.Generator) -> Circuit: ...


def swap_count(n_edges: int, fraction: float) -> int:
    """Number of successful double-edge swaps for a given rewiring fraction.

    Each swap touches two edges, hence the division: `fraction = 1.0` means
    every edge is expected to be touched once, not that the graph is fully
    randomised.
    """
    return int(round(fraction * n_edges / 2))


def edge_overlap(a: Circuit, b: Circuit) -> float:
    """Fraction of a's (pre, post) pairs still present in b."""
    pairs_a = set(zip(a.pre.tolist(), a.post.tolist()))
    pairs_b = set(zip(b.pre.tolist(), b.post.tolist()))
    return len(pairs_a & pairs_b) / len(pairs_a)


class DegreePreservingRewire:
    """Double-edge swap: (a->b), (c->d) becomes (a->d), (c->b).

    Exactly preserves every node's in- and out-degree. `fraction` controls
    the number of SUCCESSFUL swaps: round(fraction * n_edges / 2). Swaps that
    would create a self-loop or a duplicate edge are rejected and retried.
    """

    name = "degree_preserving_rewire"

    def apply(self, c: Circuit, fraction: float,
              rng: np.random.Generator) -> Circuit:
        pre, post = c.pre.copy(), c.post.copy()
        target = swap_count(c.n_edges, fraction)
        if target == 0:
            return replace(c, pre=pre, post=post)

        pairs = set(zip(pre.tolist(), post.tolist()))
        done, attempts, max_attempts = 0, 0, target * 100
        while done < target and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, c.n_edges, size=2)
            if i == j:
                continue
            new_i, new_j = (pre[i], post[j]), (pre[j], post[i])
            if new_i[0] == new_i[1] or new_j[0] == new_j[1]:
                continue
            if new_i in pairs or new_j in pairs:
                continue
            pairs.discard((pre[i], post[i]))
            pairs.discard((pre[j], post[j]))
            post[i], post[j] = post[j], post[i]
            pairs.add(new_i)
            pairs.add(new_j)
            done += 1

        if done < target:
            raise RuntimeError(
                f"only {done}/{target} swaps succeeded in {attempts} attempts")
        return replace(c, pre=pre, post=post)


class WeightShuffle:
    """Perturbation B -- reserved. Permutes synapse strengths."""

    name = "weight_shuffle"

    def apply(self, c: Circuit, fraction: float,
              rng: np.random.Generator) -> Circuit:
        raise NotImplementedError("Perturbation B is not part of this round")


class SignShuffle:
    """Perturbation C -- reserved. Permutes excitatory/inhibitory labels."""

    name = "sign_shuffle"

    def apply(self, c: Circuit, fraction: float,
              rng: np.random.Generator) -> Circuit:
        raise NotImplementedError("Perturbation C is not part of this round")
