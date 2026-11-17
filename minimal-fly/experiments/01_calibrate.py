"""Calibration on seeds A only.

1. Noise amplitude: free-membrane SD of 2 mV (exact, no simulation).
2. Tonic MN drive: an unconnected MN fires at 40 Hz.
3. ORN gain: an ORN fires at 50 Hz at concentration K.
4. Synapse count n: the smallest value in the grid at which the baseline
   circuit reaches the food in >= 80% of 200 trials on seeds A. If none
   does, the antenna separation steps up to 5 mm once; if still none, the
   baseline is reported as failed.

Writes minifly/frozen.py and results/01_calibration.json.
"""
import time

from minifly import analysis, calibrate, circuit, inference, sim
from minifly import config as cfg
from minifly.neuron import noise_sigma_for_sd


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "01_calibrate.log")
    t0 = time.time()
    sigma = noise_sigma_for_sd(cfg.NOISE_SD_MV)
    log(f"noise sigma = {sigma:.6f} mV/sqrt(ms) for a {cfg.NOISE_SD_MV} mV free-membrane SD")
    mn_drive, mn_rate = calibrate.drive_for_rate(cfg.BASE_MN_RATE_HZ, sigma, key=1)
    log(f"MN drive = {mn_drive:.6f} mV/ms -> {mn_rate:.2f} Hz")
    orn_half, orn_rate = calibrate.drive_for_rate(cfg.ORN_RATE_AT_K_HZ, sigma, key=2)
    orn_gain = 2.0 * orn_half
    log(f"ORN gain A = {orn_gain:.6f} mV/ms -> {orn_rate:.2f} Hz at C = K")
    drives = sim.Drives(mn_drive=mn_drive, orn_gain=orn_gain)

    record = {"seeds": "A (stream %d)" % cfg.STREAM_CALIBRATION,
              "noise_sigma": sigma, "mn_drive": mn_drive, "mn_rate_hz": mn_rate,
              "orn_gain": orn_gain, "orn_rate_at_k_hz": orn_rate, "grids": []}
    chosen_n, chosen_sep = None, None
    for sep in (cfg.ANTENNA_SEP_MM, cfg.FALLBACK_ANTENNA_SEP_MM):
        conds = [sim.Condition(f"n={n}", circuit.weight_matrix(
                     circuit.edges_of(circuit.BASELINE), n), circuit.thresholds(),
                     antenna_sep_mm=sep) for n in cfg.N_SYN_GRID]
        log(f"baseline grid at antenna separation {sep} mm ...")
        res = sim.parallel_closed_loop(conds, list(range(cfg.N_TRIALS)),
                                       cfg.STREAM_CALIBRATION, drives,
                                       n_workers=n_workers, log=log)
        rows = []
        for i, n in enumerate(cfg.N_SYN_GRID):
            k = int((res.status[i] == 1).sum())
            rows.append({"n_syn": n, "successes": k, "success_rate": k / cfg.N_TRIALS,
                         "success_ci": list(inference.wilson(k, cfg.N_TRIALS))})
            log(f"  n = {n:4d}: {k}/{cfg.N_TRIALS} reached the food")
        record["grids"].append({"antenna_sep_mm": sep, "rows": rows})
        passing = [r["n_syn"] for r in rows
                   if r["success_rate"] >= cfg.CALIBRATION_SUCCESS_TARGET]
        if passing:
            chosen_n, chosen_sep = min(passing), sep
            break

    passed = chosen_n is not None
    if not passed:
        # Registered fallback exhausted: freeze the largest grid value so later
        # scripts can still run, but every result is exploratory.
        chosen_n, chosen_sep = cfg.N_SYN_GRID[-1], cfg.FALLBACK_ANTENNA_SEP_MM
    record.update({"n_syn": chosen_n, "antenna_sep_mm": chosen_sep,
                   "baseline_passed": passed})
    log(f"frozen: n = {chosen_n}, antenna separation = {chosen_sep} mm, "
        f"baseline passed = {passed} ({(time.time() - t0) / 60:.1f} min)")

    cfg.FROZEN_PATH.write_text(
        '"""Calibrated values, written by\n'
        'experiments/01_calibrate.py from seeds A only, and frozen before any\n'
        'run on seeds B. Do not edit by hand."""\n'
        f"NOISE_SIGMA = {sigma!r}\n"
        f"MN_DRIVE = {mn_drive!r}\n"
        f"ORN_GAIN = {orn_gain!r}\n"
        f"N_SYN = {chosen_n!r}\n"
        f"ANTENNA_SEP_MM = {chosen_sep!r}\n"
        f"BASELINE_PASSED = {passed!r}\n")
    analysis.write_json(record, cfg.RESULTS_DIR / "01_calibration.json")


if __name__ == "__main__":
    main()
