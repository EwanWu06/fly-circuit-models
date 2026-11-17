"""v2 exploratory analyses.

1. Sensitivity: the first 300 random circuits with interneuron synapses
   halved; Q1 repeated (BH over 300).
2. The designed 20-neuron circuit at antenna separations 0.3-5 mm (200 trials
   each): does pooling seven ORNs per side help at a real fly's scale?
"""
import importlib.util

from minifly import analysis, inference, sim, v2
from minifly import config as cfg


def _load_08():
    """Reuse the registered Q1 analysis from 08_random_circuits.py."""
    spec = importlib.util.spec_from_file_location(
        "random_circuits", cfg.ROOT / "experiments" / "08_random_circuits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "09_v2_explore.log")
    rc = _load_08()
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)

    circuits = [(f"R{k:04d}", v2.random_circuit(k)) for k in range(v2.N_SENSITIVITY)]
    conds = rc.conditions(frozen, circuits, int_scale=0.5)
    log(f"sensitivity: {len(circuits)} circuits, interneuron synapses halved")
    res = sim.parallel_closed_loop(conds, list(range(v2.N_TRIALS)), v2.STREAM_TRIALS,
                                   drives, n_workers=n_workers, group_size=30, log=log,
                                   layout=v2.LAYOUT)
    q = rc.food_finding(res, v2.N_SENSITIVITY)

    edges, _ = v2.designed_circuit()
    w = v2.weight_matrix(edges, frozen["N_SYN"])
    seps = [sim.Condition(f"D20 sep={s:g}", w, v2.thresholds(), s) for s in v2.SEPS_MM]
    log("designed 20-neuron circuit across antenna separations ...")
    sres = sim.parallel_closed_loop(seps, list(range(v2.N_SEP_TRIALS)), v2.STREAM_TRIALS,
                                    drives, n_workers=n_workers, group_size=1, log=log,
                                    layout=v2.LAYOUT)
    sep_rows = []
    for i, s in enumerate(v2.SEPS_MM):
        k = int((sres.status[i] == 1).sum())
        sep_rows.append({"sep_mm": s, "successes": k, "n": v2.N_SEP_TRIALS,
                         "ci": list(inference.wilson(k, v2.N_SEP_TRIALS))})
        log(f"  D20 at {s:g} mm: {k}/{v2.N_SEP_TRIALS}")
    out = {
        "sensitivity_half_synapses": {
            "n_circuits": v2.N_SENSITIVITY, "finds_food": q["n_finds"],
            "fraction": q["fraction"], "ci": q["ci"], "worse": q["n_worse"],
            "wording": q["wording"]},
        "designed_by_separation": sep_rows,
    }
    log(f"sensitivity: {q['n_finds']}/{v2.N_SENSITIVITY} find food")
    analysis.write_json(out, cfg.RESULTS_DIR / "09_v2_explore.json")


if __name__ == "__main__":
    main()
