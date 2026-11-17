"""Calibrate sigma against B1, freeze it, then evaluate B1-B4.

The pass criteria live in src/criteria.py and were fixed before this
script was first executed. Its first run was fully deterministic
and exposed a structural defect, not a result: with no per-neuron noise,
the visual input layer fires in perfect lockstep, so the giant fiber's
pre-spike window is identically flat and max_depolarisation_mv /
m3_peak_response_time_ms read 0.0 everywhere. That deterministic run is preserved at
results/01_benchmarks_deterministic.json and is reported alongside this
one, not replaced by it.

This version desynchronises the visual input layer with
stimulus.apply_input_jitter (src/config.INPUT_JITTER_MS), pre-registered
before this script was changed to use it. Every
simulation in this script is driven by one explicit numpy.random.Generator
seeded from SEED, so the whole run is reproducible; SEED is never chosen
or adjusted based on a benchmark outcome.

Two additions, both specified in the design before this code was written:
evaluate_all also returns the per-GF values behind its two-GF measures, read
from the simulations it already runs; and main() ends with gain_sensitivity(),
the input-gain sensitivity analysis. Its generators are
spawned only after every existing draw, so no earlier result changes.
"""
import json

import numpy as np

from src import config, criteria, extract, lif, measure, stimulus

SEED = 20260922
ALT_SEEDS = (SEED + 1, SEED + 2, SEED + 3, SEED + 4, SEED + 5)


def _run(c, rv_ms, sigma_deg, eye="both", silence=(), *,
         rng: np.random.Generator,
         gain: float = config.INPUT_GAIN_MV_PER_MS):
    drive, t = stimulus.input_drive(c, rv_ms, config.DT_MS, sigma_deg, eye)
    drive = stimulus.apply_input_jitter(drive, c, rng)
    if silence:
        drive[c.mask_by_type(*silence)] = 0.0
    # The gain is required. At lif.simulate's 1.0 default the entire circuit
    # is silent -- 0/311 visual neurons and 0/2 giant fibers ever spike.
    return lif.simulate(c, drive, t, gain)


def calibrate_sigma(c, rng: np.random.Generator) -> float:
    """Pick the sigma whose B1 angular sizes sit closest to 40 deg."""
    target, best, best_err = 40.0, None, np.inf
    for sigma in criteria.SIGMA_SEARCH_DEG:
        thetas = [measure.m2_angular_size_at_spike_deg(
                      c, _run(c, rv, sigma, rng=rng), rv)
                  for rv in config.RV_VALUES_MS]
        if any(x is None for x in thetas):
            continue
        err = float(np.mean([(x - target) ** 2 for x in thetas]))
        if err < best_err:
            best, best_err = float(sigma), err
    if best is None:
        raise RuntimeError("no sigma produced a GF spike at any r/v")
    return best


def _per_gf(c, r) -> dict:
    """Each giant fiber's values from one simulation, keyed by side.

    first_spike_step: the GF's own first spike (None if it never spikes).
    max_depolarisation_own_window_mv: its maximum before its own first spike
    (its whole trace if it never spikes). max_depolarisation_shared_window_mv:
    its maximum in the window the two-GF measures use, before the first spike
    of either GF.
    """
    first = measure.gf_first_spike_steps(c, r)
    own = measure.gf_max_depolarisation_mv(c, r)
    shared = measure.gf_max_depolarisation_shared_window_mv(c, r)
    return {side: {"first_spike_step": first[side],
                   "max_depolarisation_own_window_mv": own[side],
                   "max_depolarisation_shared_window_mv": shared[side]}
            for side in first}


def evaluate_all(c, sigma: float, rng: np.random.Generator,
                 gain: float = config.INPUT_GAIN_MV_PER_MS) -> dict:
    """Compute the raw B1-B4 measurements and verdicts at a fixed sigma.

    Shared by main() (the primary, pre-registered seed), seed_stability()
    (alternate seeds) and gain_sensitivity() (other input gains), so none of
    them can compute B1-B4 differently. `gain` defaults to the frozen
    INPUT_GAIN_MV_PER_MS; only gain_sensitivity() passes another value.
    """
    theta_by_rv, peak_by_rv = {}, {}
    lc4_only, lplc2_only = {}, {}
    per_gf_intact, per_gf_lc4_only, per_gf_lplc2_only = {}, {}, {}
    for rv in config.RV_VALUES_MS:
        r = _run(c, rv, sigma, rng=rng, gain=gain)
        theta_by_rv[rv] = measure.m2_angular_size_at_spike_deg(c, r, rv)
        peak_by_rv[rv] = measure.m3_peak_response_time_ms(c, r)
        r_lc4_only = _run(c, rv, sigma, silence=("LPLC2",), rng=rng, gain=gain)
        lc4_only[rv] = measure.max_depolarisation_mv(c, r_lc4_only)
        r_lplc2_only = _run(c, rv, sigma, silence=("LC4",), rng=rng, gain=gain)
        lplc2_only[rv] = measure.max_depolarisation_mv(c, r_lplc2_only)
        per_gf_intact[rv] = _per_gf(c, r)
        per_gf_lc4_only[rv] = _per_gf(c, r_lc4_only)
        per_gf_lplc2_only[rv] = _per_gf(c, r_lplc2_only)

    rv_mid = 40.0
    r_silenced = _run(c, rv_mid, sigma, silence=("LPLC2",), rng=rng, gain=gain)
    peak_silenced = measure.m3_peak_response_time_ms(c, r_silenced)
    r_both = _run(c, rv_mid, sigma, "both", rng=rng, gain=gain)
    r_left = _run(c, rv_mid, sigma, "left", rng=rng, gain=gain)
    r_right = _run(c, rv_mid, sigma, "right", rng=rng, gain=gain)
    index = measure.m4_binocular_index(r_both, r_left, r_right, c)

    return {
        "theta_at_spike_by_rv": theta_by_rv,
        "peak_time_by_rv": peak_by_rv,
        "lc4_contribution_by_rv": lc4_only,
        "lplc2_contribution_by_rv": lplc2_only,
        "binocular_index": index,
        "peak_time_lplc2_silenced_ms": peak_silenced,
        "max_depolarisation_lplc2_silenced_mv":
            measure.max_depolarisation_mv(c, r_silenced),
        "binocular_depolarisation_mv": {
            "both": measure.max_depolarisation_mv(c, r_both),
            "left": measure.max_depolarisation_mv(c, r_left),
            "right": measure.max_depolarisation_mv(c, r_right),
        },
        "B1_passed": bool(criteria.check_b1(theta_by_rv)),
        "B2_passed": bool(criteria.check_b2(lc4_only, lplc2_only)),
        "B3_passed": bool(criteria.check_b3(peak_by_rv[rv_mid], peak_silenced)),
        "B4_passed": bool(criteria.check_b4(index)),
        # Per-GF values from the same simulations as the measures above.
        "per_gf": {
            "intact_by_rv": per_gf_intact,
            "lc4_only_by_rv": per_gf_lc4_only,
            "lplc2_only_by_rv": per_gf_lplc2_only,
            "b3_lplc2_silenced": _per_gf(c, r_silenced),
            "b4": {"both": _per_gf(c, r_both),
                   "left": _per_gf(c, r_left),
                   "right": _per_gf(c, r_right)},
        },
    }


def sensitivity(c, sigma_star: float, rng: np.random.Generator) -> dict:
    """Re-evaluate B1-B4 at sigma_star +/- 20% and +/- 40%.

    sigma is never re-fitted; this only reports robustness. Each row gets
    its own independent generator, spawned from `rng`, so the table varies
    sigma alone -- not sigma confounded with a shared, continuously
    advancing noise realisation (a fresh generator's draws are unrelated
    to how far the caller's rng has already been advanced by
    calibrate_sigma/evaluate_all).
    """
    factors = (0.6, 0.8, 1.0, 1.2, 1.4)
    row_rngs = rng.spawn(len(factors))
    rows = {}
    for factor, row_rng in zip(factors, row_rngs):
        sigma = sigma_star * factor
        theta = {rv: measure.m2_angular_size_at_spike_deg(
                     c, _run(c, rv, sigma, rng=row_rng), rv)
                 for rv in config.RV_VALUES_MS}
        lc4 = {rv: measure.max_depolarisation_mv(
                   c, _run(c, rv, sigma, silence=("LPLC2",), rng=row_rng))
               for rv in config.RV_VALUES_MS}
        lplc2 = {rv: measure.max_depolarisation_mv(
                     c, _run(c, rv, sigma, silence=("LC4",), rng=row_rng))
                 for rv in config.RV_VALUES_MS}
        index = measure.m4_binocular_index(
            _run(c, 40.0, sigma, "both", rng=row_rng),
            _run(c, 40.0, sigma, "left", rng=row_rng),
            _run(c, 40.0, sigma, "right", rng=row_rng), c)
        rows[f"{factor:.1f}x"] = {
            "sigma_deg": sigma,
            "B1_passed": bool(criteria.check_b1(theta)),
            "B2_passed": bool(criteria.check_b2(lc4, lplc2)),
            "B4_passed": bool(criteria.check_b4(index)),
        }
    return rows


def seed_stability(c, sigma: float) -> dict:
    """Re-evaluate B1-B4 at the frozen, calibrated sigma under five
    alternate seeds for the input-jitter generator.

    sigma is never re-fitted and SEED itself is never reused here -- each
    alternate seed gets its own independent generator, so this reports
    whether a verdict is a property of this regime or an artefact of one
    particular random draw.
    """
    rows = {}
    for seed in ALT_SEEDS:
        result = evaluate_all(c, sigma, np.random.default_rng(seed))
        rows[str(seed)] = {k: result[k] for k in
                            ("B1_passed", "B2_passed", "B3_passed", "B4_passed")}
    return rows


def gain_sensitivity(c, sigma: float, rng: np.random.Generator) -> dict:
    """Re-run the full B1-B4 evaluation at 0.6-1.4x the input gain.

    Specified in the design before this function existed. sigma stays frozen at the
    calibrated value; the gain is factor x INPUT_GAIN_MV_PER_MS for the sigma
    table's own factors, and each row is evaluate_all() in full at that gain,
    per-GF values included. Each row gets its own generator, spawned from
    `rng`; main() calls this after every existing use of `rng`, so no earlier
    value changes. Nothing is refitted, and INPUT_GAIN_MV_PER_MS stays 8.0.
    """
    factors = (0.6, 0.8, 1.0, 1.2, 1.4)
    row_rngs = rng.spawn(len(factors))
    rows = {}
    for factor, row_rng in zip(factors, row_rngs):
        gain = config.INPUT_GAIN_MV_PER_MS * factor
        rows[f"{factor:.1f}x"] = {"gain_mv_per_ms": gain, "sigma_deg": sigma,
                                  **evaluate_all(c, sigma, row_rng, gain=gain)}
    return rows


def main() -> None:
    c = extract.build_circuit()
    rng = np.random.default_rng(SEED)
    sigma = calibrate_sigma(c, rng)

    out = {"sigma_deg": sigma}
    out.update(evaluate_all(c, sigma, rng))
    out["sigma_sensitivity"] = sensitivity(c, sigma, rng)
    out["seed_sensitivity"] = seed_stability(c, sigma)
    # Last, so its spawned generators come after every existing draw.
    out["gain_sensitivity"] = gain_sensitivity(c, sigma, rng)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # No default=str: every value must already be JSON-native.
    (config.RESULTS_DIR / "01_benchmarks.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    from src import figures
    figures.plot_benchmarks(out, config.RESULTS_DIR / "fig1_benchmarks.png")


if __name__ == "__main__":
    main()
