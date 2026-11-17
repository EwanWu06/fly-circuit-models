"""Shared helpers for the experiment scripts: frozen values, conditions,
summaries and JSON output."""
import importlib
import json
from math import exp

import numpy as np

from . import circuit, inference
from . import config as cfg
from . import metrics, sim


def load_frozen() -> dict:
    """Calibrated values. Refuses to run before calibration has been frozen."""
    if not cfg.FROZEN_PATH.exists():
        raise RuntimeError("minifly/frozen.py is missing: run experiments/01_calibrate.py")
    frozen = importlib.import_module("minifly.frozen")
    importlib.reload(frozen)
    return {k: getattr(frozen, k) for k in
            ("NOISE_SIGMA", "MN_DRIVE", "ORN_GAIN", "N_SYN", "ANTENNA_SEP_MM",
             "BASELINE_PASSED")}


def drives(frozen: dict) -> sim.Drives:
    return sim.Drives(mn_drive=frozen["MN_DRIVE"], orn_gain=frozen["ORN_GAIN"])


def condition(name, edges, frozen, v_th=None, sep=None, noise_sd=None) -> sim.Condition:
    return sim.Condition(
        name, circuit.weight_matrix(edges, frozen["N_SYN"]),
        circuit.thresholds() if v_th is None else v_th,
        antenna_sep_mm=frozen["ANTENNA_SEP_MM"] if sep is None else sep,
        noise_sd_mv=cfg.NOISE_SD_MV if noise_sd is None else noise_sd)


def steer_levels(gain: float, sep: float):
    """Tethered mirror stimuli: both ORNs at
    concentration K for the pre-period, then the left-right difference a fly
    at the start radius meets when side-on to the food. Returns the levels
    for the left-stronger and right-stronger stimulus, each (2, 2)."""
    delta = cfg.START_RADIUS_MM * sep / (2.0 * cfg.ODOR_SIGMA_MM ** 2)
    k = cfg.ODOR_K
    drive = lambda c: gain * c / (c + k)
    base = drive(k)
    hi, lo = drive(k * exp(delta)), drive(k * exp(-delta))
    left = [[base, base], [hi, lo]]
    right = [[base, base], [lo, hi]]
    return left, right, delta


def summarize(res, c: int, seed: int) -> dict:
    m = metrics.trial_metrics(res, c)
    n = m["success"].size
    k = int(m["success"].sum())
    lo, hi = inference.wilson(k, n)
    t_ok = m["time_s"][m["success"]]
    eff = m["efficiency"][m["success"]]
    out = {
        "name": res.names[c], "n": n, "successes": k, "success_rate": k / n,
        "success_ci": [lo, hi],
        "lost_rate": float(m["lost"].mean()), "timeout_rate": float(m["timeout"].mean()),
        "time_to_food_median_s": float(np.median(t_ok)) if k else None,
        "time_to_food_iqr_s": [float(np.percentile(t_ok, 25)), float(np.percentile(t_ok, 75))] if k else None,
        "path_efficiency_median": float(np.median(eff)) if k else None,
        "approach_index_mean": float(m["approach"].mean()),
        "approach_index_ci": list(inference.bootstrap_mean_ci(m["approach"], cfg.N_BOOT, seed)),
        "speed_mean_mm_s": float(m["speed"].mean()),
        "speed_ci": list(inference.bootstrap_mean_ci(m["speed"], cfg.N_BOOT, seed + 1)),
    }
    if res.traj is not None:
        circ = np.array([metrics.is_circling(res.traj[c, i]) for i in range(n)])
        out["circling_rate"] = float(circ.mean())
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def write_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False) + "\n")


def logger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        fh.write(msg + "\n")
    return log
