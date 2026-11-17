"""Sensitivity to the two most influential design values.
Exploratory. The frozen synapse count is kept; P0 and the blind walker P3b
are re-run at each setting with the same 200 starts of seeds B.

S1 antenna separation 0.3-5 mm. S2 noise SD 1, 2, 4 mV.
"""
import numpy as np

from minifly import analysis, circuit, sim
from minifly import config as cfg


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "05_sensitivity.log")
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    base = circuit.edges_of(circuit.BASELINE)
    blind = circuit.edges_of(circuit.drop(circuit.BASELINE, circuit.CROSSED_MOTOR_EDGES))
    conds = []
    for sep in cfg.S1_ANTENNA_SEPS_MM:
        conds += [analysis.condition(f"S1 P0 sep={sep:g}", base, frozen, sep=sep),
                  analysis.condition(f"S1 P3b sep={sep:g}", blind, frozen, sep=sep)]
    for sd in cfg.S2_NOISE_SDS_MV:
        conds += [analysis.condition(f"S2 P0 sd={sd:g}", base, frozen, noise_sd=sd),
                  analysis.condition(f"S2 P3b sd={sd:g}", blind, frozen, noise_sd=sd)]
    res = sim.parallel_closed_loop(conds, list(range(cfg.N_TRIALS)), cfg.STREAM_MAIN,
                                   drives, n_workers=n_workers, group_size=1, log=log)
    summ = {n: analysis.summarize(res, i, seed=cfg.SEED_BASE + 500 + 10 * i)
            for i, n in enumerate(res.names)}
    s1 = [summ[f"S1 P0 sep={s:g}"]["success_rate"] for s in cfg.S1_ANTENNA_SEPS_MM]
    s2 = [summ[f"S2 P0 sd={s:g}"]["success_rate"] for s in cfg.S2_NOISE_SDS_MV]
    out = {"conditions": summ,
           "S1_baseline_success_by_sep": dict(zip(map(str, cfg.S1_ANTENNA_SEPS_MM), s1)),
           "S1_rises_with_separation": bool(np.all(np.diff(s1) >= 0)),
           "S2_baseline_success_by_noise": dict(zip(map(str, cfg.S2_NOISE_SDS_MV), s2)),
           "S2_falls_with_noise": bool(np.all(np.diff(s2) <= 0))}
    for n in res.names:
        log(f"  {n:18s} {summ[n]['successes']:3d}/200")
    analysis.write_json(out, cfg.RESULTS_DIR / "05_sensitivity.json")


if __name__ == "__main__":
    main()
