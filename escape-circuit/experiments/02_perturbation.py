"""Graded degree-preserving rewiring: N=100 nulls per gradient level.

sigma is read from the benchmark results and never recalibrated -- the
perturbed circuits are driven by exactly the same stimulus as the real one,
input jitter included: one jittered
drive is drawn once and reused for every circuit, so the null distribution
varies the wiring alone.
"""
import json

import numpy as np

from src import config, extract, lif, measure, perturb, stats, stimulus

FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)
N_REPEATS = 100
RV_MS = 40.0
SEED = 20260922
ALT_SEEDS = (SEED + 1, SEED + 2, SEED + 3, SEED + 4, SEED + 5)


def _streams(seed: int) -> list[np.random.Generator]:
    """Independent (jitter, rewire) generators: neither one's draws depend
    on how far the other has advanced."""
    return np.random.default_rng(seed).spawn(2)


def _measures(c, drive, t) -> dict[str, float]:
    r = lif.simulate(c, drive, t, config.INPUT_GAIN_MV_PER_MS)
    theta = measure.m2_angular_size_at_spike_deg(c, r, RV_MS)
    return {
        "theta_at_spike_deg": np.nan if theta is None else theta,
        "peak_time_ms": measure.m3_peak_response_time_ms(c, r),
        "max_depolarisation_mv": measure.max_depolarisation_mv(c, r),
        "gf_spike_rate": measure.gf_spike_rate(c, r),
    }


def main() -> None:
    sigma = json.loads(
        (config.RESULTS_DIR / "01_benchmarks.json").read_text())["sigma_deg"]
    c = extract.build_circuit()
    jitter_rng, rewire_rng = _streams(SEED)
    # Rewiring permutes edges, never neurons, so these input rows are the
    # same neurons in every rewired circuit.
    base, t = stimulus.input_drive(c, RV_MS, config.DT_MS, sigma, "both")
    drive = stimulus.apply_input_jitter(base, c, jitter_rng)
    observed = _measures(c, drive, t)
    rewire = perturb.DegreePreservingRewire()

    out = {"sigma_deg": sigma, "rv_ms": RV_MS, "seed": SEED,
           "input_jitter_ms": config.INPUT_JITTER_MS,
           "observed": observed,
           # The real circuit under the jitter draw each alternate seed would
           # use: how much conditioning on one draw matters.
           "observed_by_alt_seed": {
               str(s): _measures(c, stimulus.apply_input_jitter(
                   base, c, _streams(s)[0]), t)
               for s in ALT_SEEDS},
           "levels": {}}

    for fraction in FRACTIONS:
        nulls: dict[str, list[float]] = {k: [] for k in observed}
        overlaps = []
        for _ in range(N_REPEATS):
            shuffled = rewire.apply(c, fraction, rewire_rng)
            overlaps.append(perturb.edge_overlap(c, shuffled))
            for key, value in _measures(shuffled, drive, t).items():
                nulls[key].append(value)

        level = {"edge_overlap_mean": float(np.mean(overlaps)), "measures": {}}
        for key, values in nulls.items():
            arr = np.array(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            level["measures"][key] = {
                "null_mean": float(np.nanmean(arr)),
                "null_sd": float(np.nanstd(arr, ddof=1)),
                "n_finite": int(finite.size),
                "z": stats.z_score(observed[key], finite) if finite.size > 1
                     else float("nan"),
                "p": stats.empirical_p(observed[key], finite) if finite.size > 1
                     else float("nan"),
                "null": arr.tolist(),
            }
        out["levels"][str(fraction)] = level
        print(f"f={fraction}: overlap={level['edge_overlap_mean']:.3f}")

    flat = [(f, k, lv["measures"][k]["p"])
            for f, lv in out["levels"].items() for k in observed]
    p_values = np.array([p for _, _, p in flat], dtype=float)
    valid = np.isfinite(p_values)
    rejected = np.zeros(p_values.size, dtype=bool)
    rejected[valid] = stats.bh_fdr(p_values[valid])
    out["fdr"] = [{"fraction": f, "measure": k, "p": p,
                   "significant": bool(rejected[i])}
                  for i, (f, k, p) in enumerate(flat)]

    (config.RESULTS_DIR / "02_perturbation.json").write_text(
        json.dumps(out, indent=2))

    from src import figures
    figures.plot_dose_response(out, config.RESULTS_DIR / "fig2_dose_response.png")
    figures.plot_convergence(out, config.RESULTS_DIR / "fig3_convergence.png")


if __name__ == "__main__":
    main()
