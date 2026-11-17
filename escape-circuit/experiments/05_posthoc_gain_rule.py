"""POST HOC re-check of the rule that fixed INPUT_GAIN_MV_PER_MS.

POST HOC. It is NOT part of the pre-registration, and it changes nothing:
INPUT_GAIN_MV_PER_MS stays 8.0.

The design fixes the gain as the smallest gain, in steps of
1.0, at which the fastest condition (r/v = 10 ms) recruits the entire
311-neuron visual input layer. src/config.py records the counts measured when
it was fixed: 3.0 -> 186, 5.0 -> 293, 6.0 -> 304, 8.0 -> 311. The rule
predates the input jitter, and its outcome depends on sigma, which it does
not name. This script recounts the visual neurons that spike at least once at
r/v = 10 ms, with no jitter, for gains 1.0 to 9.0 in steps of 1.0, at
sigma = 15.0 deg (the width the stimulus tests use, which reproduces the
recorded counts) and at the final, calibrated sigma of 5.0 deg. It reports
the smallest gain that recruits every visual neuron.
"""
import json

from src import config, extract, lif, stimulus

RV_MS = 10.0
GAINS = tuple(float(g) for g in range(1, 10))
SIGMAS_DEG = (15.0, 5.0)
# The counts recorded in the comment above INPUT_GAIN_MV_PER_MS in src/config.py.
RECORDED_IN_CONFIG = {"3.0": 186, "5.0": 293, "6.0": 304, "8.0": 311}


def main() -> None:
    c = extract.build_circuit()
    visual = c.mask_by_type("LC4", "LPLC2")
    n_visual = int(visual.sum())
    out = {
        "post_hoc": True,
        "note": ("POST HOC re-check; not part of the "
                 "pre-registration. Changes nothing. "
                 "See the script's docstring."),
        "rv_ms": RV_MS,
        "input_jitter": False,
        "n_visual": n_visual,
        "gain_in_use_mv_per_ms": config.INPUT_GAIN_MV_PER_MS,
        "recorded_counts_in_config": RECORDED_IN_CONFIG,
        "recruited_by_sigma": {},
        "smallest_gain_recruiting_all_by_sigma": {},
    }
    for sigma in SIGMAS_DEG:
        drive, t = stimulus.input_drive(c, RV_MS, config.DT_MS, sigma, "both")
        counts = {}
        for gain in GAINS:
            r = lif.simulate(c, drive, t, gain)
            counts[str(gain)] = int(r.spikes[visual].any(axis=1).sum())
        full = [g for g in GAINS if counts[str(g)] == n_visual]
        out["recruited_by_sigma"][str(sigma)] = counts
        out["smallest_gain_recruiting_all_by_sigma"][str(sigma)] = (
            min(full) if full else None)
        print(f"sigma {sigma}: {counts}")
    out["recorded_counts_reproduced_at_sigma_15"] = all(
        out["recruited_by_sigma"]["15.0"][g] == n
        for g, n in RECORDED_IN_CONFIG.items())
    (config.RESULTS_DIR / "05_posthoc_gain_rule.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
