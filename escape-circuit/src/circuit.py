"""The extracted subcircuit: nodes, signed edges, and serialisation."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Circuit:
    body_ids: np.ndarray   # (N,) int64
    types: np.ndarray      # (N,) str
    sides: np.ndarray      # (N,) str, 'L' or 'R'
    pre: np.ndarray        # (E,) int32, index into body_ids
    post: np.ndarray       # (E,) int32
    weight: np.ndarray     # (E,) float64, signed synapse count

    @property
    def n_neurons(self) -> int:
        return int(self.body_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.pre.size)

    def index_of(self, body_id: int) -> int:
        hits = np.flatnonzero(self.body_ids == body_id)
        if hits.size != 1:
            raise KeyError(f"body_id {body_id} not found")
        return int(hits[0])

    def mask_by_type(self, *types: str) -> np.ndarray:
        return np.isin(self.types, np.asarray(types))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, body_ids=self.body_ids, types=self.types, sides=self.sides,
            pre=self.pre, post=self.post, weight=self.weight,
        )

    @classmethod
    def load(cls, path: Path) -> "Circuit":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in
                      ("body_ids", "types", "sides", "pre", "post", "weight")})
