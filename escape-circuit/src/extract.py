"""Extract the LC4/LPLC2 -> inhibitory relay -> DNp01 escape subcircuit."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as ft

from . import config, download
from .circuit import Circuit

INPUT_TYPES = ("LC4", "LPLC2")
GF_TYPE = "DNp01"


def _load_tables():
    paths = download.fetch_all()
    ann = ft.read_table(paths["annotations"]).to_pandas()
    nt = (ft.read_table(paths["neurotransmitters"]).to_pandas()
          .drop_duplicates("body").set_index("body"))
    weights = ft.read_table(paths["weights"], memory_map=True)
    return ann, nt, weights


def _edges_into(weights, targets: np.ndarray) -> pd.DataFrame:
    mask = pc.is_in(weights["body_post"],
                    value_set=pa.array(targets.tolist(), type=pa.int64()))
    return weights.filter(mask).to_pandas()


def _edges_between(weights, src: np.ndarray, dst: np.ndarray) -> pd.DataFrame:
    mask = pc.and_(
        pc.is_in(weights["body_pre"],
                 value_set=pa.array(src.tolist(), type=pa.int64())),
        pc.is_in(weights["body_post"],
                 value_set=pa.array(dst.tolist(), type=pa.int64())),
    )
    return weights.filter(mask).to_pandas()


def build_circuit(stats_out: dict | None = None) -> Circuit:
    """Build the subcircuit. Pass `stats_out` to receive extraction counts."""
    ann, nt, weights = _load_tables()
    ann = ann[ann["status"].astype(str).str.contains("Traced", na=False)
              & ann["type"].notna()]
    type_str = ann["type"].astype(str)

    gf = ann.loc[type_str == GF_TYPE, "bodyId"].to_numpy()
    lc = ann.loc[type_str.isin(INPUT_TYPES), "bodyId"].to_numpy()

    # Inhibitory neurons that synapse onto GF above threshold.
    onto_gf = _edges_into(weights, gf)
    onto_gf = onto_gf[onto_gf["weight"] >= config.MIN_SYNAPSE_WEIGHT]
    sign = onto_gf["body_pre"].map(nt["consensus_nt"]).map(config.SIGN_BY_NT)
    inhibitory = onto_gf.loc[sign == -1.0, "body_pre"].unique()

    # Keep only those actually driven by the visual input layer.
    lc_to_inh = _edges_between(weights, lc, inhibitory)
    lc_to_inh = lc_to_inh[lc_to_inh["weight"] >= config.MIN_SYNAPSE_WEIGHT]
    relays = np.intersect1d(lc_to_inh["body_post"].unique(), inhibitory)

    body_ids = np.unique(np.concatenate([lc, relays, gf]))
    keep = set(body_ids.tolist())

    edges = _edges_between(weights, body_ids, body_ids)
    edges = edges[edges["weight"] >= config.MIN_SYNAPSE_WEIGHT]

    signs = edges["body_pre"].map(nt["consensus_nt"]).map(config.SIGN_BY_NT)
    n_before = len(edges)
    edges = edges[signs.notna()]
    signs = signs[signs.notna()]
    if stats_out is not None:
        stats_out["edges_before_nt_filter"] = int(n_before)
        stats_out["edges_dropped_untypeable"] = int(n_before - len(edges))
        stats_out["dropped_fraction"] = float(
            (n_before - len(edges)) / n_before) if n_before else 0.0

    pos = {b: i for i, b in enumerate(body_ids.tolist())}
    meta = ann.set_index("bodyId")
    return Circuit(
        body_ids=body_ids.astype(np.int64),
        types=meta.loc[body_ids, "type"].astype(str).to_numpy().astype("<U16"),
        sides=meta.loc[body_ids, "somaSide"].astype(str).str[:1]
                  .to_numpy().astype("<U1"),
        pre=edges["body_pre"].map(pos).to_numpy().astype(np.int32),
        post=edges["body_post"].map(pos).to_numpy().astype(np.int32),
        weight=(edges["weight"].to_numpy() * signs.to_numpy()).astype(np.float64),
    )
