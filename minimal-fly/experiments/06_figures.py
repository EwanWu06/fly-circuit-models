"""All result figures, from the JSON/NPZ files the experiments wrote."""
from minifly import analysis, figures
from minifly import config as cfg


def main() -> None:
    out = cfg.RESULTS_DIR
    frozen = analysis.load_frozen()
    figures.trajectories(out / "fig1_trajectories.png")
    figures.rasters(out / "fig2_rasters.png")
    figures.outcomes(out / "fig3_outcomes.png")
    figures.rewiring(out / "fig4_rewiring.png")
    figures.tethered(out / "fig5_tethered.png")
    figures.sensitivity(out / "fig6_sensitivity.png", frozen["ANTENNA_SEP_MM"])
    if (out / "08_random_circuits.json").exists():
        figures.random_circuits(out / "fig7_random_circuits.png")
        figures.v2_trajectories(out / "fig8_v2_trajectories.png")
    if (out / "10_posthoc_v2_mechanism.json").exists():
        figures.posthoc_turning(out / "fig9_posthoc_turning.png")


if __name__ == "__main__":
    main()
