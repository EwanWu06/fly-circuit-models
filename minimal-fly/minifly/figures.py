"""Figures. Thin marks, solid hairline grid, text in ink
tones, categorical hues in a fixed order (blue, orange), and a neutral gray
for "no effect". All labels are in English."""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from . import config as cfg  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CLASS_COLORS = {"approach": BLUE, "avoid": ORANGE, "neither": MUTED}
DEG = 180.0 / np.pi

LABELS = {
    "P0": "Baseline (designed circuit)",
    "P1": "P1  crossed pathway flipped",
    "P3a": "P3a  no in-brain comparison",
    "P3b": "P3b  blind: no sensory-to-motor path",
    "P2+6": "P2  all thresholds +6 mV",
    "P2s+6": "P2′  sensory thresholds +6 mV",
}


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def _ygrid(ax):
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def _arena(ax, lim=64.0):
    ax.add_patch(Circle((0, 0), cfg.FOOD_RADIUS_MM, color=AQUA, alpha=0.30, lw=0))
    ax.add_patch(Circle((0, 0), cfg.FOOD_RADIUS_MM, fill=False, ec=AQUA, lw=1.0))
    for r in (cfg.START_RADIUS_MM, cfg.LOST_RADIUS_MM):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=0.8))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def _load(name):
    return json.loads((cfg.RESULTS_DIR / name).read_text())


def checkpoints(out_path, cp):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    ax = axes[0]
    ax.plot(cp["lif"]["t_ms"], cp["lif"]["v_mv"], color=BLUE, lw=1.4)
    ax.axhline(cfg.LIF["v_th"], color=INK2, lw=0.8)
    ax.text(cp["lif"]["t_ms"][-1], cfg.LIF["v_th"] + 0.3, "threshold", ha="right",
            fontsize=8, color=INK2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane potential (mV)")
    ax.set_title("1  One MN: integrate, fire, reset")
    _ygrid(ax)

    ax = axes[1]
    o = cp["odour"]
    ax.plot(o["distance_mm"], o["left"], color=BLUE, label="left antenna (food side)")
    ax.plot(o["distance_mm"], o["right"], color=ORANGE, label="right antenna")
    ax.set_yscale("log")
    ax.set_xlabel("distance from food (mm)")
    ax.set_ylabel("odour concentration")
    ax.set_title(f"2  Side-on to the food ({o['sep_mm']:g} mm apart)")
    ax.legend(loc="lower left")
    _ygrid(ax)

    ax = axes[2]
    for key, color in (("straight", BLUE), ("turn", ORANGE)):
        p = cp["body"][key]
        ax.plot(p[:, 0], p[:, 1], color=color)
        ax.plot(p[-1, 0], p[-1, 1], "o", ms=5, color=color, mec=SURFACE, mew=1.2)
    ax.plot(0, 0, "o", ms=5, color=MUTED, mec="none")
    ax.text(19.5, -1.2, "legs equal (40/40 Hz): straight", ha="right", va="top",
            fontsize=8, color=INK2)
    ax.text(5.0, 6.5, "right legs faster (35/45 Hz):\nturns left", ha="left",
            fontsize=8, color=INK2)
    ax.set_ylim(-4.0, 10.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("4  Two seconds of walking")
    _ygrid(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def trajectories(out_path, names=("P0", "P1", "P3a", "P3b", "P2+6", "P2s+6")):
    setup()
    data = np.load(cfg.RESULTS_DIR / "02_trajectories.npz")
    main = _load("02_main.json")["conditions"]
    all_names = list(data["names"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for ax, name in zip(axes.flat, names):
        c = all_names.index(name)
        _arena(ax)
        for t in range(data["traj"].shape[1]):
            xy = data["traj"][c, t]
            ok = ~np.isnan(xy[:, 0])
            ax.plot(xy[ok, 0], xy[ok, 1], color=BLUE, lw=0.9, alpha=0.55)
            ax.plot(xy[0, 0], xy[0, 1], "o", ms=2.5, color=MUTED, mec="none")
        s = main[name]
        ax.set_title(LABELS[name], fontsize=9.5)
        ax.text(0.0, 1.0, f"reached food {s['successes']}/{s['n']}\n"
                f"speed {s['speed_mean_mm_s']:.1f} mm/s",
                transform=ax.transAxes, color=INK2, fontsize=8, va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2})
    axes.flat[0].text(0, 0, "food", ha="center", va="center", fontsize=7, color=INK)
    axes.flat[0].text(0, cfg.START_RADIUS_MM + 1.5, "start circle (30 mm)",
                      ha="center", fontsize=7, color=MUTED)
    fig.suptitle("First 20 of 200 trials per condition (same starts in every panel)",
                 x=0.02, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def rasters(out_path, names=("P0", "P1", "P3a", "P3b"), window_s=2.0):
    setup()
    spk = np.load(cfg.RESULTS_DIR / "02_spikes.npz")
    trj = np.load(cfg.RESULTS_DIR / "02_trajectories.npz")
    all_names = list(spk["names"])
    fig, axes = plt.subplots(len(names), 2, figsize=(10.0, 2.3 * len(names)),
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    for row, name in enumerate(names):
        c = all_names.index(name)
        sel = (spk["spikes"][:, 0] == c) & (spk["spikes"][:, 1] == 0)
        steps, neurons = spk["spikes"][sel, 2], spk["spikes"][sel, 3]
        t = steps * cfg.DT_S
        ax = axes[row, 0]
        for j in range(cfg.N_NEURONS):
            tj = t[(neurons == j) & (t <= window_s)]
            ax.vlines(tj, j - 0.36, j + 0.36, color=INK2, lw=0.6)
        ax.set_yticks(range(cfg.N_NEURONS))
        ax.set_yticklabels(cfg.NEURONS)
        ax.set_ylim(cfg.N_NEURONS - 0.5, -0.5)
        ax.set_xlim(0, window_s)
        ax.set_title(f"{LABELS[name]}: spikes, trial 1, first {window_s:g} s",
                     fontsize=9)
        if row == len(names) - 1:
            ax.set_xlabel("time (s)")
        ax2 = axes[row, 1]
        _arena(ax2)
        xy = trj["traj"][c, 0]
        ok = ~np.isnan(xy[:, 0])
        ax2.plot(xy[ok, 0], xy[ok, 1], color=BLUE, lw=1.2)
        ax2.plot(xy[0, 0], xy[0, 1], "o", ms=4, color=MUTED, mec="none")
        ax2.set_title("path, whole trial", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def outcomes(out_path):
    setup()
    main = _load("02_main.json")["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    rows = ["P0", "P1", "P3a", "P3b"]
    for i, n in enumerate(rows):
        s = main[n]
        lo, hi = s["success_ci"]
        ax.hlines(i, lo, hi, color=INK2, lw=1.0)
        ax.plot(s["success_rate"], i, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5)
        ax.text(hi + 0.03, i, f"{s['successes']}/{s['n']}", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[n].replace("  ", " ") for n in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("reached the food (95% CI)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Success by wiring")

    doses = [0.0] + list(cfg.THRESHOLD_DOSES_MV)
    series = [("all 8 neurons (P2)", "P2", BLUE), ("sensory + interneurons (P2′)", "P2s", ORANGE)]
    for ax, key, ci_key, title, ylab in (
            (axes[1], "success_rate", "success_ci", "Success vs threshold dose",
             "reached the food"),
            (axes[2], "speed_mean_mm_s", "speed_ci", "Walking speed vs threshold dose",
             "mean speed (mm/s)")):
        for label, prefix, color in series:
            names = ["P0"] + [f"{prefix}+{d:g}" for d in cfg.THRESHOLD_DOSES_MV]
            y = np.array([main[n][key] for n in names])
            lo = np.array([main[n][ci_key][0] for n in names])
            hi = np.array([main[n][ci_key][1] for n in names])
            ax.fill_between(doses, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(doses, y, "-o", color=color, ms=4.5, mec=SURFACE, mew=1.2,
                    label=label)
        ax.set_xticks(doses)
        ax.set_xlabel("threshold raised by (mV)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        _ygrid(ax)
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def rewiring(out_path):
    setup()
    with open(cfg.RESULTS_DIR / "03_rewiring_wirings.csv") as fh:
        rows = list(csv.DictReader(fh))
    blind = _load("02_main.json")["conditions"]["P3b"]["success_rate"]
    bias = np.array([float(r["steering_bias_mean_rad_s"]) for r in rows]) * DEG
    succ = np.array([float(r["success_rate"]) for r in rows])
    cls = np.array([r["class"] for r in rows])
    designed = int(next(i for i, r in enumerate(rows) if r["is_designed"] == "True"))
    mirror = int(rows[designed]["mirror_index"])
    counts = {c: int(np.sum(cls == c)) for c in CLASS_COLORS}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    order = np.argsort(bias)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    for c, color in CLASS_COLORS.items():
        m = cls == c
        ax.scatter(rank[m], bias[m], s=6, color=color, lw=0,
                   label=f"{c} ({counts[c]})")
    ax.axhline(0, color=AXIS, lw=0.8)
    for i, text in ((designed, "designed"), (mirror, "its mirror (MN inputs swapped)")):
        ax.scatter([rank[i]], [bias[i]], s=60, facecolor="none", edgecolor=INK, lw=1.0)
        right = bias[i] < 0
        ax.annotate(text, (rank[i], bias[i]), xytext=(10 if right else -10, 4 if right else 8),
                    textcoords="offset points", ha="left" if right else "right",
                    fontsize=8, color=INK)
    ax.set_xlabel("964 wirings, sorted by steering bias")
    ax.set_ylabel("steering bias (deg/s, + = toward odour)")
    ax.set_title("Same degrees, every direction of steering")
    ax.legend(loc="upper left", markerscale=2)
    _ygrid(ax)

    ax = axes[1]
    for c, color in CLASS_COLORS.items():
        m = cls == c
        ax.scatter(bias[m], succ[m], s=9, color=color, lw=0, alpha=0.8, label=c)
    ax.axhline(blind, color=INK2, lw=0.8)
    ax.text(bias.min(), blind + 0.015, "blind walker (P3b)", fontsize=8, color=INK2)
    ax.scatter([bias[designed]], [succ[designed]], s=70, facecolor="none",
               edgecolor=INK, lw=1.0)
    ax.annotate("designed", (bias[designed], succ[designed]), xytext=(-10, 6),
                textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.set_xlabel("steering bias (deg/s)")
    ax.set_ylabel("reached the food (100 trials)")
    ax.set_title("Steering bias and success: turning the right way is not enough")
    ax.set_ylim(-0.02, 1.02)
    _ygrid(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def tethered(out_path):
    setup()
    g = _load("04_fly_benchmarks.json")
    tr = g["traces"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    ax = axes[0]
    t = np.arange(len(tr["g1_unilateral"])) - tr["t0_bin"]
    ax.axvspan(0, cfg.G1_STIM_MS, color=WASH, lw=0)
    ax.plot(t, np.array(tr["g1_unilateral"]) * DEG, color=BLUE, label="left ORN only")
    ax.plot(t, np.array(tr["g1_symmetric"]) * DEG, color=ORANGE, label="both ORNs equal")
    ax.axhline(0, color=AXIS, lw=0.8)
    if g["G2"]["latency_ms"] is not None:
        ax.axvline(g["G2"]["latency_ms"], color=INK2, lw=0.8)
        ax.text(g["G2"]["latency_ms"] + 8, ax.get_ylim()[1] * 0.9,
                f"turn starts {g['G2']['latency_ms']:.0f} ms\n(real fly ~70 ms)",
                fontsize=8, color=INK2, va="top")
    ax.set_xlabel("time from stimulus onset (ms)")
    ax.set_ylabel("turning (deg/s, + = left)")
    ax.set_title("G1  one antenna vs both (body fixed)")
    ax.legend(loc="upper left")
    _ygrid(ax)

    ax = axes[1]
    t3 = np.arange(len(tr["g3"])) - tr["g3_t0_bin"]
    ax.axvspan(0, cfg.G3_PULSE_MS, color=WASH, lw=0)
    ax.plot(t3, np.array(tr["g3"]) * DEG, color=BLUE)
    ax.axhline(0, color=AXIS, lw=0.8)
    lo, hi = (v * DEG for v in g["G3"]["ci"])
    ax.set_xlabel("time from pulse onset (ms)")
    ax.set_ylabel("turning (deg/s, + = left)")
    ax.set_title(f"G3  left 5% stronger for 50 ms: mean {g['G3']['mean'] * DEG:+.2f} deg/s "
                 f"[{lo:+.2f}, {hi:+.2f}]", fontsize=9)
    _ygrid(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def random_circuits(out_path):
    """v2: each random circuit against its own uniform-odour control, and
    steering bias against the crossed index."""
    setup()
    with open(cfg.RESULTS_DIR / "08_random_circuits.csv") as fh:
        rows = list(csv.DictReader(fh))
    summ = _load("08_random_circuits.json")
    rand = [r for r in rows if r["finds_food"] != ""]
    d20 = next(r for r in rows if r["name"] == "D20")
    succ = np.array([float(r["success"]) for r in rand])
    ctrl = np.array([float(r["control"]) for r in rand])
    finds = np.array([r["finds_food"] == "True" for r in rand])
    worse = np.array([r["worse"] == "True" for r in rand])
    sb = np.array([float(r["steering_deg_s"]) for r in rand])
    xi = np.array([int(r["crossed_index"]) for r in rand])
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    jitter = lambda n: rng.uniform(-0.004, 0.004, n)
    for mask, color, label in ((~finds & ~worse, MUTED, f"no difference ({int((~finds & ~worse).sum())})"),
                               (worse, ORANGE, f"worse than control ({int(worse.sum())})"),
                               (finds, BLUE, f"finds food ({int(finds.sum())})")):
        ax.scatter(ctrl[mask] + jitter(mask.sum()), succ[mask] + jitter(mask.sum()),
                   s=10, color=color, lw=0, alpha=0.8, label=label)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    ax.scatter([float(d20["control"])], [float(d20["success"])], s=70, facecolor="none",
               edgecolor=INK, lw=1.0)
    ax.annotate("designed 20-neuron circuit", (float(d20["control"]), float(d20["success"])),
                xytext=(8, -4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("reached food, uniform odour (own control)")
    ax.set_ylabel("reached food, odour field")
    ax.set_title(f"1,000 random circuits: {summ['Q1']['finds_food']} find food")
    ax.legend(loc="upper right", markerscale=2)
    _ygrid(ax)

    ax = axes[1]
    ax.scatter(xi + rng.uniform(-0.25, 0.25, xi.size), sb, s=8, color=BLUE, lw=0, alpha=0.5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    p1 = summ["P1"]
    ax.set_xlabel("crossed index (left-right wiring by chance)")
    ax.set_ylabel("steering bias (deg/s, + = toward odour)")
    ax.set_title(f"Steering follows the chance wiring (Spearman {p1['spearman_rho']:.2f})")
    _ygrid(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def v2_trajectories(out_path):
    setup()
    data = np.load(cfg.RESULTS_DIR / "08_trajectories.npz")
    names = list(data["names"])
    with open(cfg.RESULTS_DIR / "08_random_circuits.csv") as fh:
        table = {r["name"]: r for r in csv.DictReader(fh)}
    rates = [float(table[n[:-5]]["control"]) if n.endswith("-null")
             else float(table[n]["success"]) for n in names]
    titles = []
    for i, n in enumerate(names):
        if n == "D20":
            titles.append("Designed 20-neuron circuit")
        elif n.endswith("-null"):
            titles.append(f"{n[:-5]} in uniform odour (control)")
        else:
            titles.append(f"Random circuit {n}")
    order = []
    for i, n in enumerate(names):          # put each control right after its circuit
        if not n.endswith("-null"):
            order.append(i)
            order += [j for j, m in enumerate(names) if m == n + "-null"]
    k = len(order)
    cols = 4
    rows_n = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(13.0, 3.6 * rows_n))
    for ax in axes.flat[k:]:
        ax.axis("off")
    for ax, i in zip(axes.flat, order):
        _arena(ax)
        for t in range(data["traj"].shape[1]):
            xy = data["traj"][i, t]
            ok = ~np.isnan(xy[:, 0])
            ax.plot(xy[ok, 0], xy[ok, 1], color=BLUE, lw=0.9, alpha=0.55)
            ax.plot(xy[0, 0], xy[0, 1], "o", ms=2.5, color=MUTED, mec="none")
        ax.set_title(titles[i], fontsize=9.5)
        ax.text(0.0, 1.0, f"reached food {rates[i] * 100:.0f}/100",
                transform=ax.transAxes, color=INK2, fontsize=8, va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2})
    fig.suptitle("v2: first 20 trials of each circuit (same starts in every panel)",
                 x=0.02, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def posthoc_turning(out_path, names=("R0475", "R0124", "R0284", "D20")):
    """Post hoc: turning rate against the concentration at the fly's body,
    in the odour field and in uniform odour."""
    setup()
    d = _load("10_posthoc_v2_mechanism.json")
    edges = np.array(d["concentration_bins"])
    mids = np.sqrt(np.maximum(edges[:-1], 0.02) * np.minimum(edges[1:], 1.0))
    fig, axes = plt.subplots(1, len(names), figsize=(12.5, 3.3), sharey=False)
    for ax, n in zip(axes, names):
        e = d["circuits"][n]
        for key, color, label in (("odour", BLUE, "odour field"), ("uniform", MUTED, "uniform odour (control)")):
            y = np.array(e[key]["turn_deg_s"], float)
            ok = np.array(e[key]["samples"]) >= 500
            ax.plot(mids[ok], y[ok], "-o", color=color, ms=4, mec=SURFACE, mew=1.2, label=label)
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
        ax.set_ylim(bottom=0)
        title = "Designed D20" if n == "D20" else f"Random {n}"
        ax.set_title(f"{title}: {e['success'] * 100:.0f}/100 vs {e['control'] * 100:.0f}/100", fontsize=9)
        ax.set_xlabel("odour concentration at the body")
        _ygrid(ax)
    axes[0].set_ylabel("turning rate (deg/s)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Post hoc: some random circuits turn more where the odour is stronger "
                 "(turning rate, not direction)", x=0.02, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def sensitivity(out_path, frozen_sep):
    setup()
    s = _load("05_sensitivity.json")["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for ax, tag, values, key, xlabel, title in (
            (axes[0], "S1", cfg.S1_ANTENNA_SEPS_MM, "sep", "antenna separation (mm, log scale)",
             "S1  antenna separation"),
            (axes[1], "S2", cfg.S2_NOISE_SDS_MV, "sd", "membrane noise SD (mV)",
             "S2  neuronal noise")):
        for cond, color, label in (("P0", BLUE, "baseline"), ("P3b", ORANGE, "blind walker")):
            names = [f"{tag} {cond} {key}={v:g}" for v in values]
            y = np.array([s[n]["success_rate"] for n in names])
            lo = np.array([s[n]["success_ci"][0] for n in names])
            hi = np.array([s[n]["success_ci"][1] for n in names])
            ax.fill_between(values, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(values, y, "-o", color=color, ms=4.5, mec=SURFACE, mew=1.2, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("reached the food")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        _ygrid(ax)
    axes[0].set_xscale("log")
    axes[0].set_xticks(cfg.S1_ANTENNA_SEPS_MM)
    axes[0].set_xticklabels([f"{v:g}" for v in cfg.S1_ANTENNA_SEPS_MM])
    axes[0].axvline(frozen_sep, color=INK2, lw=0.8)
    axes[0].text(frozen_sep * 0.95, 0.5, "used in\nthis study", ha="right", fontsize=8, color=INK2)
    axes[0].text(0.3 * 1.05, 0.96, "real fly scale", fontsize=8, color=INK2, va="top")
    axes[1].set_xticks(cfg.S2_NOISE_SDS_MV)
    axes[0].legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
