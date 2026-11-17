"""P4: all 964 degree-preserving wirings.

Each wiring: 100 closed-loop trials (the first 100 starts of seeds B) and 100
tethered mirror-stimulus pairs. Class from the steering bias: approach if
significantly > 0, avoid if significantly < 0, else neither (z test, BH over
all 964).

Checks (not findings):
  - the original wiring reproduces P0's first 100 trials exactly;
  - mirror pairs (MN inputs swapped) fall in opposite classes, or both in
    "neither", in >= 95% of wirings.
Exploratory: class sizes, the designed wiring's rank among approachers.
"""
import csv

import numpy as np

from minifly import analysis, circuit, inference, metrics, rewire, sim
from minifly import config as cfg

OPPOSITE = {"approach": "avoid", "avoid": "approach", "neither": "neither"}


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "03_rewiring.log")
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    wirings = rewire.enumerate_wirings()
    base_idx = rewire.index_of(wirings, circuit.edges_of(circuit.BASELINE))
    conds = [analysis.condition(f"W{i:03d}", w, frozen) for i, w in enumerate(wirings)]
    trials = list(range(cfg.N_TRIALS_REWIRING))

    log(f"closed loop: {len(conds)} wirings x {len(trials)} trials on seeds B")
    res = sim.parallel_closed_loop(conds, trials, cfg.STREAM_MAIN, drives,
                                   n_workers=n_workers, group_size=20, log=log)
    left, right, delta = analysis.steer_levels(frozen["ORN_GAIN"], frozen["ANTENNA_SEP_MM"])
    t_conds = [c for c in conds for _ in (0, 1)]
    levels = [left if k % 2 == 0 else right for k in range(len(t_conds))]
    log("tethered steering bias ...")
    tres = sim.parallel_tethered(t_conds, levels, [cfg.TETHER_PRE_MS, cfg.STEER_STIM_MS],
                                 (0.0, cfg.STEER_STIM_MS), trials, cfg.STREAM_MAIN,
                                 drives, n_workers=n_workers, group_size=40, log=log)
    sb = (tres.mean_omega[0::2] - tres.mean_omega[1::2]) / 2.0

    p = np.array([inference.z_test_two_sided(sb[i]) for i in range(len(wirings))])
    sig = inference.bh(p, cfg.ALPHA)
    sb_mean = sb.mean(axis=1)
    cls = np.where(sig & (sb_mean > 0), "approach",
                   np.where(sig & (sb_mean < 0), "avoid", "neither"))

    mirror_idx = np.array([rewire.index_of(wirings, rewire.mirror(w)) for w in wirings])
    consistent = np.array([cls[mirror_idx[i]] == OPPOSITE[cls[i]]
                           for i in range(len(wirings))])

    main_trials = np.load(cfg.RESULTS_DIR / "02_main_trials.npz")
    p0 = list(main_trials["names"]).index("P0")
    same_as_p0 = all(np.array_equal(getattr(res, f)[base_idx],
                                    main_trials[f][p0, :len(trials)])
                     for f in ("status", "end_step", "path_mm", "cos_sum", "final"))

    rows = []
    for i, w in enumerate(wirings):
        mt = metrics.trial_metrics(res, i)
        ok = mt["success"]
        rows.append({
            "index": i, "wiring": rewire.describe(w), "class": cls[i],
            "steering_bias_mean_rad_s": float(sb_mean[i]), "steering_p": float(p[i]),
            "success_rate": float(ok.mean()),
            "time_to_food_median_s": float(np.median(mt["time_s"][ok])) if ok.any() else None,
            "approach_index_mean": float(mt["approach"].mean()),
            "speed_mean_mm_s": float(mt["speed"].mean()),
            "mirror_index": int(mirror_idx[i]), "is_designed": i == base_idx,
        })

    approachers = sorted((r for r in rows if r["class"] == "approach"),
                         key=lambda r: (-r["success_rate"],
                                        r["time_to_food_median_s"] or np.inf))
    rank = next(k for k, r in enumerate(approachers, 1) if r["is_designed"]) \
        if cls[base_idx] == "approach" else None
    base_rate = rows[base_idx]["success_rate"]
    counts = {c: int(np.sum(cls == c)) for c in ("approach", "avoid", "neither")}
    summary = {
        "n_wirings": len(wirings), "designed_index": base_idx,
        "steer_log_delta": delta, "class_counts": counts,
        "checks": {
            "original_reproduces_P0_first_100_exactly": bool(same_as_p0),
            "mirror_pair_consistency": float(consistent.mean()),
            "mirror_pair_consistency_at_least_95pct": bool(consistent.mean() >= 0.95),
        },
        "designed": rows[base_idx],
        "designed_rank_among_approachers": rank,
        "approachers_with_higher_success": sum(r["success_rate"] > base_rate for r in approachers),
        "top_approachers": approachers[:10],
        "success_rate_by_class": {
            c: [r["success_rate"] for r in rows if r["class"] == c] for c in counts},
    }
    log(f"classes: {counts}; mirror consistency {consistent.mean():.3f}; "
        f"original == P0: {same_as_p0}; designed rank {rank}/{len(approachers)}")
    analysis.write_json(summary, cfg.RESULTS_DIR / "03_rewiring.json")
    with open(cfg.RESULTS_DIR / "03_rewiring_wirings.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
