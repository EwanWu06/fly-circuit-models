"""Chapter 7 checkpoints 1-4, as one figure and a JSON record.

1. LIF: one MN under its frozen tonic drive (no noise): integrate, fire, reset.
2. Odour: what each antenna smells, side-on to the food, at each distance.
3. Wiring: where a single spike from each neuron lands 1.8 ms later.
4. Body: equal legs walk straight; faster right legs turn left.
"""
import numpy as np

from minifly import analysis, body, circuit, environment as env, figures
from minifly import config as cfg
from minifly.neuron import DELAY_STEPS, Network


def lif_trace(drive, ms=150.0):
    net = Network(np.zeros((1, 1, 1)), [[cfg.LIF["v_th"]]], [0.0], [0])
    d = np.full((1, 1), drive)
    v, spikes = [], []
    for s in range(int(round(ms / cfg.DT_MS))):
        v.append(net.v[0, 0])
        if net.step(s, d)[0, 0]:
            spikes.append(s * cfg.DT_MS)
    return np.arange(len(v)) * cfg.DT_MS, np.array(v), spikes


def single_spikes(n_syn):
    w = circuit.weight_matrix(circuit.edges_of(circuit.BASELINE), n_syn)
    table = []
    for src in range(cfg.N_NEURONS):
        net = Network([w], [circuit.thresholds()], [0.0], [0])
        net.v[0, src] = cfg.LIF["v_th"] + 1.0
        net.step(0, np.zeros((1, cfg.N_NEURONS)))
        queued = net.pending[DELAY_STEPS % net.pending.shape[0], 0]
        table.append({"from": cfg.NEURONS[src], "after_ms": DELAY_STEPS * cfg.DT_MS,
                      "lands_on": {cfg.NEURONS[j]: round(float(queued[j]), 3)
                                   for j in np.flatnonzero(queued)}})
    return table


def walk(rate_left, rate_right, seconds=2.0):
    rate = np.array([[rate_left, rate_right]]) / 1000.0
    x, y, th = np.zeros(1), np.zeros(1), np.zeros(1)
    path = []
    for s in range(int(seconds / cfg.DT_S)):
        if s % 100 == 0:
            path.append((x[0], y[0]))
        fwd, omega = body.speeds(rate)
        body.advance(x, y, th, np.cos(th), np.sin(th), fwd, omega)
    return np.array(path)


def main() -> None:
    frozen = analysis.load_frozen()
    t, v, spikes = lif_trace(frozen["MN_DRIVE"])
    r = np.linspace(6.0, 60.0, 109)
    lx, ly, rx, ry = env.antenna_positions(r, 0 * r, np.zeros_like(r), np.ones_like(r),
                                           frozen["ANTENNA_SEP_MM"])
    c_left, c_right = env.concentration(lx, ly), env.concentration(rx, ry)
    cp = {
        "lif": {"t_ms": t, "v_mv": v, "spike_ms": spikes, "drive": frozen["MN_DRIVE"],
                "rate_hz_without_noise": 1000.0 * (len(spikes) - 1) / (spikes[-1] - spikes[0])},
        "odour": {"distance_mm": r, "left": c_left, "right": c_right,
                  "sep_mm": frozen["ANTENNA_SEP_MM"]},
        "single_spikes": single_spikes(frozen["N_SYN"]),
        "body": {"straight": walk(40.0, 40.0), "turn": walk(35.0, 45.0)},
    }
    figures.checkpoints(cfg.RESULTS_DIR / "fig0_checkpoints.png", cp)
    cp["lif"] = {k: cp["lif"][k] for k in ("spike_ms", "drive", "rate_hz_without_noise")}
    cp["odour"] = {"distance_mm": r[::12], "left_over_right": (c_left / c_right)[::12],
                   "sep_mm": frozen["ANTENNA_SEP_MM"]}
    cp["body"] = {k: {"end_xy_mm": p[-1]} for k, p in cp["body"].items()}
    analysis.write_json(cp, cfg.RESULTS_DIR / "00_checkpoints.json")


if __name__ == "__main__":
    main()
