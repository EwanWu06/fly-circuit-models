"""Compare LC4/LPLC2 -> GF counts against Ache et al. 2019 (FAFB).

Ache et al. 2019 reconstructed the inputs to the right GF in the right
hemisphere of FAFB, a whole-brain electron-microscopy volume of an adult
female fly, and report 55 LC4 cells / 2,442 synapses and 108 LPLC2 cells /
1,366 synapses onto it. This study's connectome, MaleCNS, is male, so the
comparison is across sex, individual, dataset and reconstruction pipeline.
Agreement is evidence that the extraction pipeline is correct.
"""
import json

import numpy as np

from src import config, extract

ACHE_2019_FAFB = {"lc4_cells": 55, "lc4_synapses": 2442,
                  "lplc2_cells": 108, "lplc2_synapses": 1366}
GF_RIGHT_BODY_ID = 10001


def compute_validation() -> dict:
    """Build circuit and compute LC4/LPLC2 -> GF validation metrics.

    Returns:
        dict with keys: ache2019_fafb_reference, extraction_stats, n_neurons, n_edges,
                       lc4_cells_right, lc4_synapses_right, lplc2_cells_right,
                       lplc2_synapses_right
    """
    stats: dict = {}
    c = extract.build_circuit(stats_out=stats)
    gf = c.index_of(GF_RIGHT_BODY_ID)
    onto_gf = c.post == gf
    out = {"ache2019_fafb_reference": ACHE_2019_FAFB, "extraction_stats": stats,
           "n_neurons": c.n_neurons, "n_edges": c.n_edges}
    for label, cell_type in (("lc4", "LC4"), ("lplc2", "LPLC2")):
        is_type = c.mask_by_type(cell_type)[c.pre] & onto_gf
        out[f"{label}_cells_right"] = int(np.unique(c.pre[is_type]).size)
        out[f"{label}_synapses_right"] = int(np.abs(c.weight[is_type]).sum())
    return out


def main() -> None:
    out = compute_validation()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "00_extraction_validation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
