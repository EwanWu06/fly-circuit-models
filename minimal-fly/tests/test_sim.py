"""Chapter 7, checkpoint 5: the closed loop runs, and is reproducible."""
import numpy as np

from minifly import circuit, config as cfg, sim

DRIVES = sim.Drives(mn_drive=2.5, orn_gain=6.0)     # test values, not calibrated
SHORT = 3000                                        # 0.3 s


def _cond(name, named, n_syn=80):
    return sim.Condition(name, circuit.weight_matrix(circuit.edges_of(named), n_syn),
                         circuit.thresholds())


P0 = _cond("P0", circuit.BASELINE)
P3B = _cond("P3b", circuit.drop(circuit.BASELINE, circuit.CROSSED_MOTOR_EDGES))


def _same(a, b, ia, ib):
    for f in ("status", "end_step", "path_mm", "cos_sum", "final"):
        assert np.array_equal(getattr(a, f)[ia], getattr(b, f)[ib]), f


def test_a_trial_does_not_depend_on_what_else_is_in_the_batch():
    alone = sim.run_closed_loop([P0], [0, 1, 2], cfg.STREAM_MAIN, DRIVES, max_steps=SHORT)
    mixed = sim.run_closed_loop([P3B, P0], [0, 1, 2, 3, 150], cfg.STREAM_MAIN,
                                DRIVES, max_steps=SHORT)
    _same(alone, mixed, (0, slice(0, 3)), (1, slice(0, 3)))
    sub = sim.run_closed_loop([P0], [2], cfg.STREAM_MAIN, DRIVES, max_steps=SHORT)
    _same(alone, sub, (0, slice(2, 3)), (0, slice(0, 1)))


def test_runs_are_deterministic_and_streams_differ():
    a = sim.run_closed_loop([P0], [0, 1], cfg.STREAM_MAIN, DRIVES, max_steps=SHORT)
    b = sim.run_closed_loop([P0], [0, 1], cfg.STREAM_MAIN, DRIVES, max_steps=SHORT)
    c = sim.run_closed_loop([P0], [0, 1], cfg.STREAM_CALIBRATION, DRIVES, max_steps=SHORT)
    _same(a, b, 0, 0)
    assert not np.array_equal(a.final, c.final)
    assert np.all(a.status == 3)              # nobody arrives in 0.3 s


def test_trajectory_starts_on_the_start_circle_and_walks():
    res = sim.run_closed_loop([P0], [0], cfg.STREAM_MAIN, DRIVES, record_traj=True,
                              n_spike_trials=1, max_steps=SHORT)
    first = res.traj[0, 0, 0]
    assert np.isclose(np.hypot(first[0], first[1]), cfg.START_RADIUS_MM)
    assert res.path_mm[0, 0] > 0
    assert res.spikes.shape[0] > 0


def test_blind_walker_turns_identically_whatever_it_smells():
    """P3b has no path from the ORNs to the legs, so mirror-image stimuli
    must give exactly the same turning, trial by trial."""
    a = DRIVES.orn_gain
    levels = [[[a / 2, a / 2], [0.6 * a, 0.4 * a]], [[a / 2, a / 2], [0.4 * a, 0.6 * a]]]
    res = sim.run_tethered([P3B, P3B], levels, [50.0, 100.0], (0.0, 100.0),
                           [0, 1, 2], cfg.STREAM_MAIN, DRIVES)
    assert np.array_equal(res.mean_omega[0], res.mean_omega[1])
    res0 = sim.run_tethered([P0, P0], levels, [50.0, 100.0], (0.0, 100.0),
                            [0, 1, 2], cfg.STREAM_MAIN, DRIVES)
    assert not np.array_equal(res0.mean_omega[0], res0.mean_omega[1])


def test_circling_detector():
    from minifly.metrics import is_circling
    t = np.arange(2000) * 0.01
    spin = np.column_stack([0.1 * np.cos(t), 0.1 * np.sin(t), t])     # 1 rad/s in place
    straight = np.column_stack([t, 0 * t, 0 * t])
    assert is_circling(spin) and not is_circling(straight)
    jitter = np.column_stack([0 * t, 0 * t, 3 * np.sin(5 * t)])        # back and forth
    assert not is_circling(jitter)
