import numpy as np
import pytest

from src import config, extract, stimulus


def test_angular_size_is_monotonic_and_hits_endpoints():
    t = stimulus.time_grid(rv_ms=40.0, dt_ms=config.DT_MS)
    theta = stimulus.angular_size_deg(t, rv_ms=40.0)
    assert np.all(np.diff(theta) > 0)
    assert theta[0] == pytest.approx(config.THETA_START_DEG, abs=0.5)
    assert theta[-1] == pytest.approx(config.THETA_END_DEG, abs=0.5)


def test_faster_approach_means_shorter_stimulus():
    """Smaller r/v = more abrupt expansion = less time from 10 to 63 deg."""
    # np.ptp(arr), not arr.ptp() -- the method was removed in NumPy 2.0.
    durations = [float(np.ptp(stimulus.time_grid(rv, config.DT_MS)))
                 for rv in config.RV_VALUES_MS]
    assert durations == sorted(durations)


def test_drive_shape_and_targets():
    c = extract.build_circuit()
    drive, t = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    assert drive.shape == (c.n_neurons, t.size)
    driven = c.mask_by_type("LC4", "LPLC2")
    assert np.all(drive[~driven] == 0.0), "only LC4/LPLC2 receive drive"
    assert drive[driven].max() > 0.0


def test_monocular_drives_only_one_side():
    c = extract.build_circuit()
    drive, _ = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "left")
    right_inputs = c.mask_by_type("LC4", "LPLC2") & (c.sides == "R")
    assert np.all(drive[right_inputs] == 0.0)
    left_inputs = c.mask_by_type("LC4", "LPLC2") & (c.sides == "L")
    assert drive[left_inputs].max() > 0.0


def test_lplc2_peaks_near_42_degrees():
    c = extract.build_circuit()
    drive, t = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    theta = stimulus.angular_size_deg(t, 40.0)
    lplc2 = c.mask_by_type("LPLC2")
    peak_theta = theta[drive[lplc2].mean(axis=0).argmax()]
    assert 38.0 <= peak_theta <= 46.0


def test_lc4_peak_drive_is_cross_condition_scaled():
    """LC4's peak drive is 8x higher at r/v=10 than at r/v=80.

    Benchmark B2 requires that LC4's contribution grows as r/v decreases
    (i.e., as the looming approach is more abrupt). The scaling is against
    the fixed reference (peak velocity of r/v=10 ms), not per-condition,
    so that this 8x dynamic range is preserved.
    """
    c = extract.build_circuit()
    lc4 = c.mask_by_type("LC4")

    # Measure LC4's peak drive across all r/v conditions
    peak_drives = []
    for rv in config.RV_VALUES_MS:
        drive, _ = stimulus.input_drive(c, rv, config.DT_MS, 15.0, "both")
        peak = drive[lc4].max()  # Peak across all LC4 neurons and time
        peak_drives.append(peak)

    # Peak drives must be strictly decreasing (as r/v increases, approach is slower)
    assert peak_drives == sorted(peak_drives, reverse=True)

    # Ratio between fastest (r/v=10) and slowest (r/v=80) should be ~8x
    ratio = peak_drives[0] / peak_drives[-1]
    assert 7.5 <= ratio <= 8.5


def test_jitter_leaves_non_visual_rows_byte_identical():
    """Relay and GF rows never receive drive, so jitter must not touch them."""
    c = extract.build_circuit()
    drive, _ = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    rng = np.random.default_rng(0)
    jittered = stimulus.apply_input_jitter(drive, c, rng)
    non_visual = ~c.mask_by_type("LC4", "LPLC2")
    assert np.array_equal(jittered[non_visual], drive[non_visual])


def test_jitter_desynchronises_visual_rows():
    """Within a single cell type, rows must not all be identical after
    jitter -- if they were, every neuron of that type would still be
    perfectly synchronised, which is exactly the defect the mechanism
    exists to eliminate.

    Comparing LC4 rows against LPLC2 rows does not test this: the two
    types carry structurally different signals (linear angular velocity
    vs. a Gaussian size envelope), so they are already non-identical with
    zero jitter applied. Each type must be checked against its own first
    row, separately.
    """
    c = extract.build_circuit()
    drive, _ = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    rng = np.random.default_rng(0)
    jittered = stimulus.apply_input_jitter(drive, c, rng)
    for cell_type in ("LC4", "LPLC2"):
        rows = jittered[c.mask_by_type(cell_type)]
        assert not np.all(rows == rows[0]), (
            f"all {cell_type} rows are identical to each other -- "
            "that population is still perfectly synchronised")


def test_jitter_is_reproducible_with_same_seed():
    c = extract.build_circuit()
    drive, _ = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    a = stimulus.apply_input_jitter(drive, c, np.random.default_rng(42))
    b = stimulus.apply_input_jitter(drive, c, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_jitter_does_not_mutate_input():
    c = extract.build_circuit()
    drive, _ = stimulus.input_drive(c, 40.0, config.DT_MS, 15.0, "both")
    original = drive.copy()
    stimulus.apply_input_jitter(drive, c, np.random.default_rng(0))
    assert np.array_equal(drive, original)
