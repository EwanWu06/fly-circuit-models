"""All plotting. Figure labels are English (see the language policy)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEASURE_LABELS = {
    "theta_at_spike_deg": "Angular size at GF spike (deg)",
    "peak_time_ms": "Peak response time (ms)",
    "max_depolarisation_mv": "Max depolarisation (mV)",
    "gf_spike_rate": "GF spike rate",
}


def plot_benchmarks(results: dict, out_path: Path) -> None:
    # Keys are floats in memory and strings after a JSON round-trip.
    theta_by_rv = {float(k): v
                   for k, v in results["theta_at_spike_by_rv"].items()}
    rvs = sorted(theta_by_rv)
    theta = [theta_by_rv[rv] for rv in rvs]
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.axhspan(30, 50, color="0.9", label="B1 pass band (30-50 deg)")
    ax.axhline(40, color="0.5", ls="--", lw=1, label="Published threshold")
    ax.plot(rvs, theta, "o-", color="#c0392b", label="Model")
    ax.set_xlabel("r/v (ms)")
    ax.set_ylabel(MEASURE_LABELS["theta_at_spike_deg"])
    ax.set_title("B1: angular size at giant fiber spike")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dose_response(results: dict, out_path: Path) -> None:
    fractions = sorted(float(f) for f in results["levels"])
    keys = list(MEASURE_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for ax, key in zip(axes.ravel(), keys):
        means, los, his = [], [], []
        for f in fractions:
            null = np.array(results["levels"][str(f)]["measures"][key]["null"])
            null = null[np.isfinite(null)]
            means.append(null.mean() if null.size else np.nan)
            lo, hi = (np.percentile(null, [2.5, 97.5]) if null.size > 1
                      else (np.nan, np.nan))
            los.append(lo)
            his.append(hi)
        ax.fill_between(fractions, los, his, alpha=0.25, color="#2980b9",
                        label="Rewired 95% interval")
        ax.plot(fractions, means, "o-", color="#2980b9", label="Rewired mean")
        ax.axhline(results["observed"][key], color="#c0392b", lw=2,
                   label="Real connectome")
        ax.set_xlabel("Rewiring dose f (f·|E|/2 swaps)")
        ax.set_ylabel(MEASURE_LABELS[key])
        ax.legend(fontsize=7)
    fig.suptitle("Dose-response to degree-preserving rewiring")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_convergence(results: dict, out_path: Path) -> None:
    fractions = sorted(float(f) for f in results["levels"])
    overlap = [results["levels"][str(f)]["edge_overlap_mean"] for f in fractions]
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.plot(fractions, overlap, "o-", color="#16a085")
    ax.set_xlabel("Rewiring dose f (f·|E|/2 swaps)")
    ax.set_ylabel("Edge overlap with real circuit")
    ax.set_title("Randomisation convergence check")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
