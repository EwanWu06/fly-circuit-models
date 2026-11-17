"""Real-fly comparison: tethered tests after Gaudry et al. 2013.
Body fixed; the two ORN drives are set directly.

G1  one ORN at the concentration-K drive for 500 ms, the other at 0: mean
    turning over 0-500 ms toward the stimulated side (CI > 0). Both equal:
    CI contains 0. (Design check.)
G2  when turning starts after onset in G1 (descriptive only).
G3  both at the K drive; the left one 5% higher for 50 ms: mean turning over
    0-300 ms to the left (CI > 0). No prediction.
"""
import numpy as np

from minifly import analysis, circuit, inference, sim
from minifly import config as cfg

RAD_TO_DEG = 180.0 / np.pi


def latency_ms(trace, t0_bin, baseline_bins=200, hold_bins=10):
    """First bin after onset where the mean trace stays above baseline
    mean + 3 SD for hold_bins consecutive 1-ms bins."""
    base = trace[t0_bin - baseline_bins:t0_bin]
    thr = base.mean() + 3.0 * base.std()
    above = trace[t0_bin:] > thr
    for b in range(above.size - hold_bins):
        if above[b:b + hold_bins].all():
            return float(b), float(thr)
    return None, float(thr)


def main() -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "04_fly_benchmarks.log")
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    p0 = analysis.condition("P0", circuit.edges_of(circuit.BASELINE), frozen)
    half = frozen["ORN_GAIN"] / 2.0          # ORN drive at concentration K
    trials = list(range(cfg.N_TRIALS))

    g1 = sim.run_tethered([p0, p0], [[[0, 0], [half, 0]], [[0, 0], [half, half]]],
                          [cfg.TETHER_PRE_MS, cfg.G1_STIM_MS], (0.0, cfg.G1_STIM_MS),
                          trials, cfg.STREAM_MAIN, drives)
    uni, sym = g1.mean_omega
    ci_uni = inference.bootstrap_mean_ci(uni, cfg.N_BOOT, cfg.SEED_BASE + 801)
    ci_sym = inference.bootstrap_mean_ci(sym, cfg.N_BOOT, cfg.SEED_BASE + 802)
    lat, thr = latency_ms(g1.trace[0], g1.t0_bin)

    g3 = sim.run_tethered([p0], [[[half, half], [half * (1 + cfg.G3_ASYMMETRY), half],
                                  [half, half]]],
                          [cfg.TETHER_PRE_MS, cfg.G3_PULSE_MS,
                           cfg.G3_WINDOW_MS - cfg.G3_PULSE_MS],
                          (0.0, cfg.G3_WINDOW_MS), trials, cfg.STREAM_MAIN, drives)
    ci_g3 = inference.bootstrap_mean_ci(g3.mean_omega[0], cfg.N_BOOT, cfg.SEED_BASE + 803)

    out = {
        "units": "turning in rad/s; + = left",
        "G1": {"unilateral_mean": float(uni.mean()), "unilateral_ci": ci_uni,
               "symmetric_mean": float(sym.mean()), "symmetric_ci": ci_sym,
               "pass": bool(ci_uni[0] > 0 and ci_sym[0] <= 0 <= ci_sym[1])},
        "G2": {"latency_ms": lat, "threshold_rad_s": thr,
               "real_fly_ms": 70.0,
               "note": "depends on the 100 ms leg smoothing; not comparable one to one"},
        "G3": {"mean": float(g3.mean_omega[0].mean()), "ci": ci_g3,
               "fraction_turning_left": float(np.mean(g3.mean_omega[0] > 0)),
               "pass": bool(ci_g3[0] > 0)},
        "traces": {"t0_bin": g1.t0_bin, "g1_unilateral": g1.trace[0],
                   "g1_symmetric": g1.trace[1], "g3": g3.trace[0],
                   "g3_t0_bin": g3.t0_bin},
    }
    log(f"G1 {'PASS' if out['G1']['pass'] else 'FAIL'}: unilateral "
        f"{uni.mean() * RAD_TO_DEG:+.1f} deg/s, symmetric {sym.mean() * RAD_TO_DEG:+.1f} deg/s")
    log(f"G2 latency: {lat} ms (real fly ~70 ms)")
    log(f"G3 {'PASS' if out['G3']['pass'] else 'FAIL'}: "
        f"{g3.mean_omega[0].mean() * RAD_TO_DEG:+.2f} deg/s, CI "
        f"[{ci_g3[0] * RAD_TO_DEG:+.2f}, {ci_g3[1] * RAD_TO_DEG:+.2f}]")
    analysis.write_json(out, cfg.RESULTS_DIR / "04_fly_benchmarks.json")


if __name__ == "__main__":
    main()
