"""POST HOC diagnostic: what the coded B2 and B4 measures read.

POST HOC. It is NOT part of the pre-registration, and nothing in it
re-scores a registered benchmark: src/criteria.py and the coded measures are unchanged,
and results/01_benchmarks.json keeps the registered verdicts. It tests one
reading of those verdicts.

The coded response R (measure.max_depolarisation_mv) averages the two giant
fibers (GFs) and stops at the first spike of either one. This script re-runs the B2 and B4
conditions so that the runs being compared share ONE input-jitter draw, and
records the coded values beside per-GF values.

- B4 at r/v = 40 ms: both eyes, left eye only and right eye only, with
  identical jitter offsets in all three runs. The offsets depend only on the
  generator and the visual rows, so a fresh numpy.random.default_rng(seed) per
  run gives identical offsets. Recorded: the coded index; the per-GF index
  over each GF's own pre-spike window; the per-GF index over the full window
  ("within the stimulus window", applied to each GF); and
  the gap between the two GFs' first spikes in the binocular run.
- B2: LC4 only and LPLC2 only at the four r/v, all eight runs sharing one
  draw, as coded values and per-GF values.
- Relay timing, from the same B4 runs: when the inhibitory relays first fire
  and when the contralateral GF first leaves rest, relative to the driven
  GF's first spike.

Seeds: SEED and the five ALT_SEEDS of experiments/01_benchmarks.py. These are
the same seed values as that script's, not the same draws: there, each seed's
generator is consumed run after run in program order.
"""
import importlib.util
import json

import numpy as np

from src import config, criteria, extract, measure

_spec = importlib.util.spec_from_file_location(
    "benchmarks", config.ROOT / "experiments" / "01_benchmarks.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)   # defines functions and constants; runs nothing

SEEDS = (_bench.SEED, *_bench.ALT_SEEDS)
B4_RV_MS = 40.0
EYES = ("both", "left", "right")
RV_KEYS = [str(rv) for rv in config.RV_VALUES_MS]


def _sim(c, rv_ms: float, sigma: float, seed: int, eye: str = "both",
         silence: tuple = ()):
    """One run of the benchmark script's _run with a fresh generator, so its
    jitter offsets depend on `seed` alone and repeat exactly across runs."""
    return _bench._run(c, rv_ms, sigma, eye, silence,
                       rng=np.random.default_rng(seed))


def _gf_rows(c) -> dict[str, int]:
    return {str(c.sides[i]): int(i) for i in measure.gf_indices(c)}


def _full_window_mv(c, r) -> dict[str, float]:
    """Each GF's maximum depolarisation over its whole trace, spikes included.
    The recorded trace is sampled before each update, so it never shows a
    value above threshold."""
    return {side: float(r.v[i].max() - config.LIF["v_0"])
            for side, i in sorted(_gf_rows(c).items())}


def _index(r_by_eye: dict, side: str) -> float:
    denominator = r_by_eye["left"][side] + r_by_eye["right"][side]
    return (float("nan") if denominator == 0.0
            else r_by_eye["both"][side] / denominator)


def _b4(c, sigma: float, seed: int) -> tuple[dict, dict]:
    runs = {eye: _sim(c, B4_RV_MS, sigma, seed, eye) for eye in EYES}
    coded_index = measure.m4_binocular_index(
        runs["both"], runs["left"], runs["right"], c)
    own = {eye: measure.gf_max_depolarisation_mv(c, r) for eye, r in runs.items()}
    full = {eye: _full_window_mv(c, r) for eye, r in runs.items()}
    first = {eye: measure.gf_first_spike_steps(c, r) for eye, r in runs.items()}
    both_first = list(first["both"].values())
    gap = (None if None in both_first
           else abs(both_first[0] - both_first[1]))
    out = {
        "coded_index": coded_index,
        "coded_B4_passed": bool(criteria.check_b4(coded_index)),
        "coded_R_mv": {eye: measure.max_depolarisation_mv(c, r)
                       for eye, r in runs.items()},
        "first_spike_step": first,
        "first_spike_gap_both_steps": gap,
        "first_spike_gap_both_ms": None if gap is None else gap * config.DT_MS,
        "own_window_mv": own,
        "own_window_index": {side: _index(own, side) for side in own["both"]},
        "shared_window_mv": {
            eye: measure.gf_max_depolarisation_shared_window_mv(c, r)
            for eye, r in runs.items()},
        "full_window_mv": full,
        "full_window_index": {side: _index(full, side) for side in full["both"]},
    }
    return out, _relay_timing(c, runs, first)


def _relay_timing(c, runs: dict, first: dict) -> dict:
    """Relay and contralateral-GF timing, in steps relative to a GF spike."""
    relays = ~c.mask_by_type("LC4", "LPLC2", measure.GF_TYPE)
    assert int(relays.sum()) == 51, int(relays.sum())
    gf = _gf_rows(c)
    v_0 = config.LIF["v_0"]
    out = {}
    for eye, r in runs.items():
        relay_steps = np.flatnonzero(r.spikes[relays].any(axis=0))
        relay_first = int(relay_steps[0]) if relay_steps.size else None
        if eye == "both":
            spikes = [s for s in first[eye].values() if s is not None]
            ref = min(spikes) if spikes else None
            out[eye] = {"first_gf_spike_step": ref,
                        "first_relay_spike_step": relay_first,
                        "relay_minus_first_gf_spike_ms": (
                            None if None in (ref, relay_first)
                            else (relay_first - ref) * config.DT_MS)}
            continue
        driven = "L" if eye == "left" else "R"
        other = "R" if driven == "L" else "L"
        ref = first[eye][driven]
        moved = np.flatnonzero(np.abs(r.v[gf[other]] - v_0) > 1e-9)
        leaves = int(moved[0]) if moved.size else None
        out[eye] = {
            "driven_gf": driven,
            "driven_gf_first_spike_step": ref,
            "first_relay_spike_step": relay_first,
            "relay_minus_driven_spike_ms": (
                None if None in (ref, relay_first)
                else (relay_first - ref) * config.DT_MS),
            "contralateral_gf_first_leaves_rest_step": leaves,
            "contralateral_leaves_rest_minus_driven_spike_ms": (
                None if None in (ref, leaves) else (leaves - ref) * config.DT_MS),
            "contralateral_gf_min_mv": float(r.v[gf[other]].min() - v_0),
            "contralateral_gf_max_mv": float(r.v[gf[other]].max() - v_0),
        }
    return out


def _b2(c, sigma: float, seed: int) -> dict:
    out: dict = {}
    for label, silence in (("lc4_only", ("LPLC2",)), ("lplc2_only", ("LC4",))):
        rows = {}
        for rv in config.RV_VALUES_MS:
            r = _sim(c, rv, sigma, seed, "both", silence)
            rows[str(rv)] = {
                "coded_mv": measure.max_depolarisation_mv(c, r),
                "theta_at_first_spike_deg":
                    measure.m2_angular_size_at_spike_deg(c, r, rv),
                "first_spike_step": measure.gf_first_spike_steps(c, r),
                "own_window_mv": measure.gf_max_depolarisation_mv(c, r),
                "shared_window_mv":
                    measure.gf_max_depolarisation_shared_window_mv(c, r),
            }
        out[label] = rows
    lc4 = {rv: out["lc4_only"][str(rv)]["coded_mv"] for rv in config.RV_VALUES_MS}
    lplc2 = np.array([out["lplc2_only"][k]["coded_mv"] for k in RV_KEYS])
    out["coded_B2_passed"] = bool(criteria.check_b2(
        lc4, {rv: out["lplc2_only"][str(rv)]["coded_mv"]
              for rv in config.RV_VALUES_MS}))
    out["coded_lc4_strictly_decreasing"] = bool(all(
        lc4[a] > lc4[b] for a, b in zip(config.RV_VALUES_MS,
                                        config.RV_VALUES_MS[1:])))
    out["coded_lplc2_cv"] = float(lplc2.std() / abs(lplc2.mean()))
    return out


def main() -> None:
    sigma = json.loads(
        (config.RESULTS_DIR / "01_benchmarks.json").read_text())["sigma_deg"]
    c = extract.build_circuit()
    out = {
        "post_hoc": True,
        "note": ("POST HOC diagnostic; not part of the "
                 "pre-registration. No registered "
                 "verdict is re-scored. See the script's docstring."),
        "sigma_deg": sigma,
        "gain_mv_per_ms": config.INPUT_GAIN_MV_PER_MS,
        "dt_ms": config.DT_MS,
        "b4_rv_ms": B4_RV_MS,
        "seeds": list(SEEDS),
        "b4_shared_draw": {},
        "relay_timing": {},
        "b2_shared_draw": {},
    }
    for seed in SEEDS:
        b4, timing = _b4(c, sigma, seed)
        out["b4_shared_draw"][str(seed)] = b4
        out["relay_timing"][str(seed)] = timing
        out["b2_shared_draw"][str(seed)] = _b2(c, sigma, seed)
        print(f"seed {seed}: coded index {b4['coded_index']:.3f}, "
              f"per-GF own-window index {b4['own_window_index']}, "
              f"first-spike gap {b4['first_spike_gap_both_steps']} steps")
    (config.RESULTS_DIR / "04_posthoc_gf_synchrony.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
