"""v2 main: 1,000 random 20-neuron circuits.

For each circuit: 100 trials in the odour field and the same 100 trials with
uniform odour (its own control), plus 100 tethered mirror-stimulus pairs.
The designed 20-neuron circuit (D20) runs alongside as a positive control.

Registered:
  Q1  share of random circuits that reach food more often than their own
      control (two-sided exact McNemar, BH over 1,000), with the wording
      fixed in advance.
  C1  approach and avoid (steering bias, BH) equally common (check).
  P1  steering bias rises with the crossed index (Spearman, one-sided).
  PC  D20 beats its own control (one-sided McNemar).
"""
import csv

import numpy as np

from minifly import analysis, inference, metrics, sim, v2
from minifly import config as cfg

WORDING = ((0.10, "a substantial share of random circuits find food on their own"),
           (0.01, "a few random circuits find food on their own"),
           (0.00, "random wiring almost never finds food on its own"))
DEG = 180.0 / np.pi


def wording(fraction: float) -> str:
    return next(text for threshold, text in WORDING if fraction >= threshold)


def conditions(frozen, circuits, int_scale: float = 1.0) -> list:
    out = []
    for name, (edges, _) in circuits:
        w = v2.weight_matrix(edges, frozen["N_SYN"], int_scale)
        sep = frozen["ANTENNA_SEP_MM"]
        out.append(sim.Condition(name, w, v2.thresholds(), sep))
        out.append(sim.Condition(name + "-null", w, v2.thresholds(), sep, uniform_odor=True))
    return out


def food_finding(res, n_random: int) -> dict:
    """Q1 on a closed-loop result whose conditions alternate odour / control."""
    succ = res.status[0::2] == 1
    ctrl = res.status[1::2] == 1
    tests = [inference.mcnemar_two_sided(succ[i], ctrl[i]) for i in range(succ.shape[0])]
    p = np.array([t[0] for t in tests])
    n10 = np.array([t[1] for t in tests])
    n01 = np.array([t[2] for t in tests])
    sig = inference.bh(p[:n_random], cfg.ALPHA)
    finds = sig & (n10[:n_random] > n01[:n_random])
    worse = sig & (n10[:n_random] < n01[:n_random])
    k = int(finds.sum())
    return {"succ": succ, "ctrl": ctrl, "p": p, "n10": n10, "n01": n01,
            "finds": finds, "worse": worse, "n_finds": k, "n_worse": int(worse.sum()),
            "fraction": k / n_random, "ci": list(inference.wilson(k, n_random)),
            "wording": wording(k / n_random)}


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "08_random_circuits.log")
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    circuits = [(f"R{k:04d}", v2.random_circuit(k)) for k in range(v2.N_CIRCUITS)]
    allc = circuits + [("D20", v2.designed_circuit())]
    conds = conditions(frozen, allc)
    trials = list(range(v2.N_TRIALS))

    log(f"closed loop: {len(allc)} circuits x 2 (odour, control) x {len(trials)} trials")
    res = sim.parallel_closed_loop(conds, trials, v2.STREAM_TRIALS, drives,
                                   n_workers=n_workers, group_size=40, log=log,
                                   layout=v2.LAYOUT)
    left, right, delta = analysis.steer_levels(frozen["ORN_GAIN"], frozen["ANTENNA_SEP_MM"])
    t_conds = [c for c in conds[0::2] for _ in (0, 1)]
    levels = [left if k % 2 == 0 else right for k in range(len(t_conds))]
    log("tethered steering bias ...")
    tres = sim.parallel_tethered(t_conds, levels, [cfg.TETHER_PRE_MS, cfg.STEER_STIM_MS],
                                 (0.0, cfg.STEER_STIM_MS), trials, v2.STREAM_TRIALS,
                                 drives, n_workers=n_workers, group_size=80, log=log,
                                 layout=v2.LAYOUT)
    sb = (tres.mean_omega[0::2] - tres.mean_omega[1::2]) / 2.0
    sbm = sb.mean(axis=1)

    n = v2.N_CIRCUITS
    q1 = food_finding(res, n)
    p_sb = np.array([inference.z_test_two_sided(sb[i]) for i in range(len(allc))])
    sig_sb = inference.bh(p_sb[:n], cfg.ALPHA)
    approach = sig_sb & (sbm[:n] > 0)
    avoid = sig_sb & (sbm[:n] < 0)
    p_c1 = inference.binomial_two_sided(int(approach.sum()), int(approach.sum() + avoid.sum()))
    x_index = np.array([v2.crossed_index(e, s) for _, (e, s) in circuits])
    rho, p_p1 = inference.spearman_greater(x_index, sbm[:n], seed=cfg.SEED_BASE + 81)
    p_pc = inference.mcnemar_greater(q1["succ"][n], q1["ctrl"][n])

    speed = np.array([metrics.trial_metrics(res, 2 * i)["speed"].mean() for i in range(len(allc))])
    succ_rate = q1["succ"].mean(axis=1)
    ctrl_rate = q1["ctrl"].mean(axis=1)
    finders = np.flatnonzero(q1["finds"])
    best = int(np.argmax(succ_rate[:n]))
    summary = {
        "seeds": f"trials stream {v2.STREAM_TRIALS}, wiring stream {v2.STREAM_WIRING}",
        "n_circuits": n, "steer_log_delta": delta,
        "Q1": {"finds_food": q1["n_finds"], "fraction": q1["fraction"], "ci": q1["ci"],
               "worse_than_control": q1["n_worse"], "wording": q1["wording"]},
        "C1": {"approach": int(approach.sum()), "avoid": int(avoid.sum()),
               "neither": int(n - approach.sum() - avoid.sum()), "p": p_c1,
               "pass": bool(p_c1 > 0.05)},
        "P1": {"spearman_rho": rho, "p_one_sided": p_p1, "pass": bool(p_p1 < 0.05)},
        "PC": {"successes": int(q1["succ"][n].sum()), "control": int(q1["ctrl"][n].sum()),
               "p": p_pc, "pass": bool(p_pc < 0.05),
               "steering_bias_deg_s": float(sbm[n] * DEG)},
        "explore": {
            "finders_that_steer_towards": int((q1["finds"] & approach).sum()),
            "finders_without_significant_steering": int((q1["finds"] & ~approach & ~avoid).sum()),
            "finders_that_steer_away": int((q1["finds"] & avoid).sum()),
            "best_random": {"name": circuits[best][0], "success": float(succ_rate[best]),
                            "control": float(ctrl_rate[best]),
                            "steering_bias_deg_s": float(sbm[best] * DEG),
                            "crossed_index": int(x_index[best]),
                            "wiring": v2.describe(circuits[best][1][0])},
            "finder_success_median": float(np.median(succ_rate[finders])) if finders.size else None,
            "control_success_median": float(np.median(ctrl_rate[:n])),
            "random_success_median": float(np.median(succ_rate[:n])),
        },
    }
    log(f"Q1: {q1['n_finds']}/{n} random circuits find food ({q1['wording']}); "
        f"C1 approach {approach.sum()} / avoid {avoid.sum()} p={p_c1:.3f}; "
        f"P1 rho={rho:.3f} p={p_p1:.4f}; PC {q1['succ'][n].sum()} vs {q1['ctrl'][n].sum()} p={p_pc:.2g}")
    analysis.write_json(summary, cfg.RESULTS_DIR / "08_random_circuits.json")

    with open(cfg.RESULTS_DIR / "08_random_circuits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "n_edges", "crossed_index", "success", "control", "n10", "n01",
                    "p_mcnemar", "finds_food", "worse", "steering_deg_s", "steering_p",
                    "steering_class", "speed_mm_s", "wiring"])
        for i, (name, (edges, signs)) in enumerate(allc):
            is_random = i < n
            cls = ("approach" if approach[i] else "avoid" if avoid[i] else "neither") \
                if is_random else ""
            w.writerow([name, len(edges), v2.crossed_index(edges, signs),
                        round(float(succ_rate[i]), 2), round(float(ctrl_rate[i]), 2),
                        int(q1["n10"][i]), int(q1["n01"][i]), float(q1["p"][i]),
                        bool(q1["finds"][i]) if is_random else "",
                        bool(q1["worse"][i]) if is_random else "",
                        round(float(sbm[i] * DEG), 2), float(p_sb[i]), cls,
                        round(float(speed[i]), 2), v2.describe(edges)])

    # Trajectories for the figures and the page: re-run a few circuits with
    # recording on. Batch invariance means the outcomes must match the main run.
    picks = [n, best]
    picks += [int(i) for i in finders[np.argsort(-succ_rate[finders])][1:3] if int(i) != best]
    picks += [int(i) for i in np.flatnonzero(~q1["finds"])[:2]]
    sel = [conds[2 * i] for i in picks] + [conds[2 * best + 1]]
    rec = sim.run_closed_loop(sel, list(range(20)), v2.STREAM_TRIALS, drives,
                              record_traj=True, layout=v2.LAYOUT)
    for j, i in enumerate(picks):
        assert np.array_equal(rec.status[j], res.status[2 * i, :20])
    np.savez_compressed(cfg.RESULTS_DIR / "08_trajectories.npz",
                        names=np.array(rec.names), traj=rec.traj[:, :, ::5].astype(np.float32),
                        status=rec.status,
                        success_rate=np.append(succ_rate[picks], ctrl_rate[best]),
                        sample_ms=np.array(5 * cfg.TRAJ_SAMPLE_STEPS * cfg.DT_MS))


if __name__ == "__main__":
    main()
