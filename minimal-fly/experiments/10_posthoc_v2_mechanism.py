"""Post hoc, not registered (written after seeing 08_random_circuits.json).

Most random circuits that find food show no left-right steering preference.
How do they do it? For each of them (and the designed D20), re-run the same
100 trials with trajectories, in the odour field and in uniform odour, and
relate walking speed and turning rate to the odour concentration at the
fly's body. If a circuit changes how it moves with concentration (kinesis)
rather than which way it turns, the relation appears in the odour field and
disappears in uniform odour, where the fly cannot sense where it is.
"""
import csv
import importlib.util

import numpy as np

from minifly import analysis, environment, sim, v2
from minifly import config as cfg

BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.325, 0.5, 0.8, 1.01])
SAMPLE_S = cfg.TRAJ_SAMPLE_STEPS * cfg.DT_S


def _load_08():
    spec = importlib.util.spec_from_file_location(
        "random_circuits", cfg.ROOT / "experiments" / "08_random_circuits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def movement_by_concentration(traj):
    """Mean speed (mm/s) and turning rate (deg/s) per concentration bin,
    from trajectories sampled every 10 ms."""
    speed_sum = np.zeros(BINS.size - 1)
    turn_sum = np.zeros(BINS.size - 1)
    count = np.zeros(BINS.size - 1)
    for t in range(traj.shape[0]):
        xy = traj[t].astype(np.float64)
        ok = ~np.isnan(xy[:, 0])
        xy = xy[ok]
        if xy.shape[0] < 3:
            continue
        mid_x = 0.5 * (xy[1:, 0] + xy[:-1, 0])
        mid_y = 0.5 * (xy[1:, 1] + xy[:-1, 1])
        conc = environment.concentration(mid_x, mid_y)
        speed = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / SAMPLE_S
        turn = np.abs(np.diff(xy[:, 2])) / SAMPLE_S * 180.0 / np.pi
        b = np.clip(np.digitize(conc, BINS) - 1, 0, BINS.size - 2)
        speed_sum += np.bincount(b, weights=speed, minlength=BINS.size - 1)
        turn_sum += np.bincount(b, weights=turn, minlength=BINS.size - 1)
        count += np.bincount(b, minlength=BINS.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return speed_sum / count, turn_sum / count, count


def main(n_workers: int = 10) -> None:
    log = analysis.logger(cfg.RESULTS_DIR / "logs" / "10_posthoc_v2_mechanism.log")
    rc = _load_08()
    frozen = analysis.load_frozen()
    drives = analysis.drives(frozen)
    with open(cfg.RESULTS_DIR / "08_random_circuits.csv") as fh:
        rows = {r["name"]: r for r in csv.DictReader(fh)}
    names = [n for n, r in rows.items() if r["finds_food"] == "True"] + ["D20"]
    circuits = [(n, v2.designed_circuit() if n == "D20" else v2.random_circuit(int(n[1:])))
                for n in names]
    conds = rc.conditions(frozen, circuits)
    res = sim.parallel_closed_loop(conds, list(range(v2.N_TRIALS)), v2.STREAM_TRIALS,
                                   drives, n_workers=n_workers, group_size=2, log=log,
                                   record_traj=True, layout=v2.LAYOUT)
    out = {"label": "post hoc, not registered", "concentration_bins": BINS.tolist(),
           "circuits": {}}
    for i, n in enumerate(names):
        succ = (res.status[2 * i] == 1).mean()
        assert np.isclose(succ, float(rows[n]["success"]))      # same trials as 08
        entry = {"success": float(succ), "control": float(rows[n]["control"]),
                 "steering_deg_s": float(rows[n]["steering_deg_s"]),
                 "steering_class": rows[n]["steering_class"]}
        for k, label in ((2 * i, "odour"), (2 * i + 1, "uniform")):
            sp, tu, cnt = movement_by_concentration(res.traj[k])
            entry[label] = {"speed": np.round(sp, 2).tolist(), "turn_deg_s": np.round(tu, 1).tolist(),
                            "samples": cnt.astype(int).tolist()}
        out["circuits"][n] = entry
        o, u = entry["odour"], entry["uniform"]
        log(f"{n}: success {succ:.2f} (control {entry['control']:.2f}); "
            f"turn by conc (odour) {o['turn_deg_s']}; (uniform) {u['turn_deg_s']}")
    analysis.write_json(out, cfg.RESULTS_DIR / "10_posthoc_v2_mechanism.json")


if __name__ == "__main__":
    main()
