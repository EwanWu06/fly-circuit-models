"""Main experiment on seeds B: P0-P3.

Closed loop: 14 conditions x 200 trials. Every condition uses the same 200
starts and noise draws, so comparisons are paired. A tethered mirror-stimulus
steering bias is also reported for each condition (descriptive).

Registered criteria (as corrected before any run):
  P1  avoids:  M4 below the blind walker P3b (paired bootstrap CI < 0) and
               success below P3b (exact McNemar, Holm).
  P2  all neurons' thresholds up: mean speed falls at every dose; success
               trend falls (Cochran-Armitage, Holm); circling < 5% at every dose.
  P2s sensory and interneurons only: success trend falls (Holm).
  P3a no in-brain comparison: M4 above P3b (paired CI > 0) and success
               above P3b (McNemar, Holm).
  P3b blind walker: the null reference, not tested.
"""
import numpy as np

from minifly import analysis, circuit, inference, metrics, sim
from minifly import config as cfg


def conditions(frozen) -> list:
    base = circuit.BASELINE
    conds = [analysis.condition("P0", circuit.edges_of(base), frozen),
             analysis.condition("P1", circuit.edges_of(
                 circuit.flip_signs(base, circuit.CROSSED_MOTOR_EDGES)), frozen)]
    for d in cfg.THRESHOLD_DOSES_MV:
        conds.append(analysis.condition(f"P2+{d:g}", circuit.edges_of(base), frozen,
                                        v_th=circuit.thresholds(d)))
    for d in cfg.THRESHOLD_DOSES_MV:
        conds.append(analysis.condition(
            f"P2s+{d:g}", circuit.edges_of(base), frozen,
            v_th=circuit.thresholds(d, circuit.SENSORY_AND_INTER)))
    conds.append(analysis.condition("P3a", circuit.edges_of(
        circuit.drop(base, circuit.COMPARISON_EDGES)), frozen))
    conds.append(analysis.condition("P3b", circuit.edges_of(
        circuit.drop(base, circuit.CROSSED_MOTOR_EDGES)), frozen))
    return conds


def steering_bias(conds, frozen, drives, trials, n_workers, log):
    left, right, delta = analysis.steer_levels(frozen["ORN_GAIN"], frozen["ANTENNA_SEP_MM"])
    t_conds = [c for c in conds for _ in (0, 1)]
    levels = [left if k % 2 == 0 else right for k in range(len(t_conds))]
    tres = sim.parallel_tethered(t_conds, levels, [cfg.TETHER_PRE_MS, cfg.STEER_STIM_MS],
                                 (0.0, cfg.STEER_STIM_MS), trials, cfg.STREAM_MAIN,
                                 drives, n_workers=n_workers, group_size=2, log=log)
    return (tres.mean_omega[0::2] - tres.mean_omega[1::2]) / 2.0, delta


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "02_main.log")
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    conds = conditions(frozen)
    trials = list(range(cfg.N_TRIALS))
    log(f"closed loop: {len(conds)} conditions x {len(trials)} trials on seeds B")
    res = sim.parallel_closed_loop(conds, trials, cfg.STREAM_MAIN, drives,
                                   n_workers=n_workers, group_size=1, log=log,
                                   record_traj=True, n_spike_trials=3)
    names = res.names
    idx = {n: i for i, n in enumerate(names)}
    m = {n: metrics.trial_metrics(res, i) for n, i in idx.items()}
    summ = {n: analysis.summarize(res, i, seed=cfg.SEED_BASE + 10 * i)
            for n, i in idx.items()}

    log("tethered steering bias ...")
    sb, delta = steering_bias(conds, frozen, drives, trials, n_workers, log)
    for n, i in idx.items():
        summ[n]["steering_bias_mean_rad_s"] = float(sb[i].mean())
        summ[n]["steering_bias_ci"] = list(inference.bootstrap_mean_ci(
            sb[i], cfg.N_BOOT, cfg.SEED_BASE + 10 * i + 2))

    # --- Registered tests -------------------------------------------------
    doses = [0.0] + list(cfg.THRESHOLD_DOSES_MV)
    p2_names = ["P0"] + [f"P2+{d:g}" for d in cfg.THRESHOLD_DOSES_MV]
    p2s_names = ["P0"] + [f"P2s+{d:g}" for d in cfg.THRESHOLD_DOSES_MV]
    totals = [cfg.N_TRIALS] * len(doses)
    p_p1 = inference.mcnemar_greater(m["P3b"]["success"], m["P1"]["success"])
    p_p2 = inference.cochran_armitage_decreasing(
        doses, [summ[n]["successes"] for n in p2_names], totals)
    p_p2s = inference.cochran_armitage_decreasing(
        doses, [summ[n]["successes"] for n in p2s_names], totals)
    p_p3a = inference.mcnemar_greater(m["P3a"]["success"], m["P3b"]["success"])
    holm_p, holm_rej = inference.holm([p_p1, p_p2, p_p2s, p_p3a], cfg.ALPHA)

    d_p1 = inference.bootstrap_mean_ci(m["P1"]["approach"] - m["P3b"]["approach"],
                                       cfg.N_BOOT, cfg.SEED_BASE + 901)
    d_p3a = inference.bootstrap_mean_ci(m["P3a"]["approach"] - m["P3b"]["approach"],
                                        cfg.N_BOOT, cfg.SEED_BASE + 902)
    speeds = [summ[n]["speed_mean_mm_s"] for n in p2_names]
    circling = {n: summ[n]["circling_rate"] for n in p2_names[1:]}

    verdicts = {
        "P1": {"m4_minus_blind_ci": d_p1, "m4_below_blind": d_p1[1] < 0,
               "success_p": p_p1, "success_p_holm": holm_p[0],
               "success_below_blind": bool(holm_rej[0]),
               "pass": bool(d_p1[1] < 0 and holm_rej[0])},
        "P2": {"speeds_by_dose": dict(zip(p2_names, speeds)),
               "speed_falls_every_dose": bool(np.all(np.diff(speeds) < 0)),
               "trend_p": p_p2, "trend_p_holm": holm_p[1],
               "trend_significant": bool(holm_rej[1]),
               "circling_rates": circling,
               "circling_below_5pct_every_dose": all(v < 0.05 for v in circling.values()),
               "pass": bool(np.all(np.diff(speeds) < 0) and holm_rej[1]
                            and all(v < 0.05 for v in circling.values()))},
        "P2s": {"trend_p": p_p2s, "trend_p_holm": holm_p[2],
                "pass": bool(holm_rej[2])},
        "P3a": {"m4_minus_blind_ci": d_p3a, "m4_above_blind": d_p3a[0] > 0,
                "success_p": p_p3a, "success_p_holm": holm_p[3],
                "success_above_blind": bool(holm_rej[3]),
                "pass": bool(d_p3a[0] > 0 and holm_rej[3])},
    }
    for k, v in verdicts.items():
        log(f"{k}: {'PASS' if v['pass'] else 'FAIL'}")
    for n in names:
        s = summ[n]
        log(f"  {n:7s} success {s['successes']:3d}/200  M4 {s['approach_index_mean']:+.3f}"
            f"  speed {s['speed_mean_mm_s']:5.2f}  SB {s['steering_bias_mean_rad_s']:+.3f}")

    analysis.write_json({
        "seeds": "B (stream %d)" % cfg.STREAM_MAIN, "frozen": frozen,
        "steer_log_delta": delta, "conditions": summ, "verdicts": verdicts,
        "baseline_passed_calibration": frozen["BASELINE_PASSED"],
    }, cfg.RESULTS_DIR / "02_main.json")

    np.savez_compressed(
        cfg.RESULTS_DIR / "02_main_trials.npz", names=np.array(names),
        status=res.status, end_step=res.end_step, path_mm=res.path_mm,
        cos_sum=res.cos_sum, final=res.final, steering_bias=sb)
    keep = 20
    np.savez_compressed(
        cfg.RESULTS_DIR / "02_trajectories.npz", names=np.array(names),
        traj=res.traj[:, :keep, ::5].astype(np.float32),
        status=res.status[:, :keep], sample_ms=np.array(5 * cfg.TRAJ_SAMPLE_STEPS * cfg.DT_MS))
    np.savez_compressed(cfg.RESULTS_DIR / "02_spikes.npz", names=np.array(names),
                        spikes=res.spikes.astype(np.int32),
                        end_step=res.end_step[:, :3])


if __name__ == "__main__":
    main()
