import numpy as np
import pytest

from src import config, lif, measure, stimulus
from src.circuit import Circuit


def _gf_only(v_trace: np.ndarray, spike_at: int | None) -> tuple:
    n, t = 1, v_trace.size
    c = Circuit(
        body_ids=np.array([10001], dtype=np.int64),
        types=np.array(["DNp01"], dtype="<U16"),
        sides=np.array(["R"], dtype="<U1"),
        pre=np.empty(0, dtype=np.int32),
        post=np.empty(0, dtype=np.int32),
        weight=np.empty(0, dtype=np.float64),
    )
    spikes = np.zeros((n, t), dtype=bool)
    if spike_at is not None:
        spikes[0, spike_at] = True
    t_ms = np.arange(t) * config.DT_MS
    return c, lif.SimResult(v=v_trace[None, :], spikes=spikes, t_ms=t_ms)


def _two_gf(v_trace: np.ndarray, spike_at: int | None) -> tuple:
    """Circuit with two DNp01 neurons, one per side."""
    n, t = 2, v_trace.size
    c = Circuit(
        body_ids=np.array([10001, 10002], dtype=np.int64),
        types=np.array(["DNp01", "DNp01"], dtype="<U16"),
        sides=np.array(["L", "R"], dtype="<U1"),
        pre=np.empty(0, dtype=np.int32),
        post=np.empty(0, dtype=np.int32),
        weight=np.empty(0, dtype=np.float64),
    )
    spikes = np.zeros((n, t), dtype=bool)
    if spike_at is not None:
        spikes[:, spike_at] = True
    t_ms = np.arange(t) * config.DT_MS
    v_expanded = np.tile(v_trace, (n, 1))
    return c, lif.SimResult(v=v_expanded, spikes=spikes, t_ms=t_ms)


def _no_gf() -> Circuit:
    """Circuit with no DNp01 neurons."""
    return Circuit(
        body_ids=np.array([5001, 5002], dtype=np.int64),
        types=np.array(["LC4", "LPLC2"], dtype="<U16"),
        sides=np.array(["L", "R"], dtype="<U1"),
        pre=np.empty(0, dtype=np.int32),
        post=np.empty(0, dtype=np.int32),
        weight=np.empty(0, dtype=np.float64),
    )


def test_m1_returns_none_when_gf_is_silent():
    c, r = _gf_only(np.full(100, -52.0), None)
    assert measure.m1_spike_latency_ms(c, r) is None


def test_m1_returns_time_of_first_spike():
    c, r = _gf_only(np.full(100, -52.0), spike_at=50)
    assert measure.m1_spike_latency_ms(c, r) == 50 * config.DT_MS


def test_m3_finds_subthreshold_peak_before_spike():
    v = np.full(100, -52.0)
    v[30] = -46.0            # subthreshold peak
    v[60] = -40.0            # would be post-spike, must be ignored
    c, r = _gf_only(v, spike_at=50)
    assert measure.m3_peak_response_time_ms(c, r) == 30 * config.DT_MS


def test_m4_is_one_for_perfectly_linear_summation():
    c, both = _gf_only(np.full(100, -52.0), None)
    both.v[0, 10] = -42.0                    # 10 mV depolarisation
    _, left = _gf_only(np.full(100, -52.0), None)
    left.v[0, 10] = -47.0                    # 5 mV
    _, right = _gf_only(np.full(100, -52.0), None)
    right.v[0, 10] = -47.0                   # 5 mV
    assert measure.m4_binocular_index(both, left, right, c) == 1.0


# A first spike at step 0 leaves an empty pre-spike window, so both measures
# raise rather than return a value.
def test_m3_raises_when_first_spike_at_step_0():
    c, r = _gf_only(np.full(100, -52.0), spike_at=0)
    with pytest.raises(ValueError, match="First GF spike at step 0"):
        measure.m3_peak_response_time_ms(c, r)


def test_max_depolarisation_raises_when_first_spike_at_step_0():
    c, r = _gf_only(np.full(100, -52.0), spike_at=0)
    with pytest.raises(ValueError, match="First GF spike at step 0"):
        measure.max_depolarisation_mv(c, r)


# A circuit without a giant fiber raises the same error in every measure.
def test_m3_raises_when_no_gf_neurons():
    c = _no_gf()
    t = np.arange(100) * config.DT_MS
    r = lif.SimResult(v=np.full((2, 100), -52.0), spikes=np.zeros((2, 100), dtype=bool), t_ms=t)
    with pytest.raises(ValueError, match="Circuit contains no DNp01"):
        measure.m3_peak_response_time_ms(c, r)


def test_max_depolarisation_raises_when_no_gf_neurons():
    c = _no_gf()
    t = np.arange(100) * config.DT_MS
    r = lif.SimResult(v=np.full((2, 100), -52.0), spikes=np.zeros((2, 100), dtype=bool), t_ms=t)
    with pytest.raises(ValueError, match="Circuit contains no DNp01"):
        measure.max_depolarisation_mv(c, r)


def test_gf_spike_rate_raises_when_no_gf_neurons():
    c = _no_gf()
    t = np.arange(100) * config.DT_MS
    r = lif.SimResult(v=np.full((2, 100), -52.0), spikes=np.zeros((2, 100), dtype=bool), t_ms=t)
    with pytest.raises(ValueError, match="Circuit contains no DNp01"):
        measure.gf_spike_rate(c, r)


def test_m2_returns_angular_size_at_spike():
    """M2 matches stimulus.angular_size_deg directly."""
    c, r = _gf_only(np.full(100, -52.0), spike_at=50)
    rv_ms = 10.0
    t_spike_ms = r.t_ms[50]
    expected = float(stimulus.angular_size_deg(t_spike_ms, rv_ms))
    assert measure.m2_angular_size_at_spike_deg(c, r, rv_ms) == expected


def test_m2_returns_none_when_gf_is_silent():
    """M2 returns None when GF never spikes."""
    c, r = _gf_only(np.full(100, -52.0), None)
    assert measure.m2_angular_size_at_spike_deg(c, r, 10.0) is None


def test_gf_spike_rate_with_two_neurons():
    """gf_spike_rate returns the fraction of GF neurons that spiked."""
    # Both neurons spike
    c, r = _two_gf(np.full(100, -52.0), spike_at=50)
    assert measure.gf_spike_rate(c, r) == 1.0

    # Only one neuron spikes
    c, r = _two_gf(np.full(100, -52.0), spike_at=None)
    spikes = np.zeros((2, 100), dtype=bool)
    spikes[0, 50] = True
    r_one_spike = lif.SimResult(v=r.v, spikes=spikes, t_ms=r.t_ms)
    assert measure.gf_spike_rate(c, r_one_spike) == 0.5

    # Neither neuron spikes
    c, r = _two_gf(np.full(100, -52.0), spike_at=None)
    assert measure.gf_spike_rate(c, r) == 0.0


def _two_gf_traces(v_left: np.ndarray, v_right: np.ndarray,
                   spike_left: int | None, spike_right: int | None) -> tuple:
    """Two DNp01 neurons, left then right, each with its own trace and spike."""
    t = v_left.size
    c = Circuit(
        body_ids=np.array([10010, 10001], dtype=np.int64),
        types=np.array(["DNp01", "DNp01"], dtype="<U16"),
        sides=np.array(["L", "R"], dtype="<U1"),
        pre=np.empty(0, dtype=np.int32),
        post=np.empty(0, dtype=np.int32),
        weight=np.empty(0, dtype=np.float64),
    )
    spikes = np.zeros((2, t), dtype=bool)
    if spike_left is not None:
        spikes[0, spike_left] = True
    if spike_right is not None:
        spikes[1, spike_right] = True
    t_ms = np.arange(t) * config.DT_MS
    return c, lif.SimResult(v=np.vstack([v_left, v_right]), spikes=spikes,
                            t_ms=t_ms)


def _staggered_pair() -> tuple:
    """Left GF peaks at 6 mV and spikes at step 40; right GF reaches 5 mV at
    step 50 and spikes at step 55, after the left one."""
    v_left = np.full(100, -52.0)
    v_left[30] = -46.0
    v_right = np.full(100, -52.0)
    v_right[50] = -47.0
    return _two_gf_traces(v_left, v_right, spike_left=40, spike_right=55)


def test_gf_first_spike_steps_are_per_gf_and_keyed_by_side():
    c, r = _staggered_pair()
    assert measure.gf_first_spike_steps(c, r) == {"L": 40, "R": 55}
    c, r = _two_gf_traces(np.full(100, -52.0), np.full(100, -52.0), 40, None)
    assert measure.gf_first_spike_steps(c, r) == {"L": 40, "R": None}


def test_gf_max_depolarisation_uses_each_gfs_own_prespike_window():
    c, r = _staggered_pair()
    # Each GF is read up to its own first spike.
    assert measure.gf_max_depolarisation_mv(c, r) == {"L": 6.0, "R": 5.0}
    # The shared window stops at the left GF's spike, before the right GF moves.
    assert measure.gf_max_depolarisation_shared_window_mv(c, r) == {
        "L": 6.0, "R": 0.0}
    # The coded measure averages the two traces in that shared window, so it
    # reports half the leading GF's value rather than either GF's response.
    assert measure.max_depolarisation_mv(c, r) == 3.0


def test_gf_max_depolarisation_reads_whole_trace_of_a_gf_that_never_spikes():
    v_left = np.full(100, -52.0)
    v_left[30] = -46.0
    v_right = np.full(100, -52.0)
    v_right[80] = -49.0            # after the left GF's spike at step 40
    c, r = _two_gf_traces(v_left, v_right, spike_left=40, spike_right=None)
    assert measure.gf_max_depolarisation_mv(c, r) == {"L": 6.0, "R": 3.0}
    assert measure.gf_max_depolarisation_shared_window_mv(c, r) == {
        "L": 6.0, "R": 0.0}


def test_gf_max_depolarisation_is_zero_for_a_gf_that_only_hyperpolarises():
    v_right = np.full(100, -52.0)
    v_right[60:] = -58.0
    c, r = _two_gf_traces(np.full(100, -52.0), v_right, None, None)
    assert measure.gf_max_depolarisation_mv(c, r) == {"L": 0.0, "R": 0.0}


def test_per_gf_helpers_raise_when_a_gf_spikes_at_step_0():
    c, r = _two_gf_traces(np.full(100, -52.0), np.full(100, -52.0), 0, 50)
    with pytest.raises(ValueError, match="First GF spike at step 0"):
        measure.gf_max_depolarisation_mv(c, r)
    with pytest.raises(ValueError, match="First GF spike at step 0"):
        measure.gf_max_depolarisation_shared_window_mv(c, r)


def test_per_gf_helpers_raise_without_one_gf_per_side():
    t = np.arange(100) * config.DT_MS
    r = lif.SimResult(v=np.full((2, 100), -52.0),
                      spikes=np.zeros((2, 100), dtype=bool), t_ms=t)
    with pytest.raises(ValueError, match="Circuit contains no DNp01"):
        measure.gf_first_spike_steps(_no_gf(), r)
    c, r = _two_gf(np.full(100, -52.0), None)
    same_side = Circuit(body_ids=c.body_ids, types=c.types,
                        sides=np.array(["L", "L"], dtype="<U1"),
                        pre=c.pre, post=c.post, weight=c.weight)
    with pytest.raises(ValueError, match="one giant fiber per side"):
        measure.gf_max_depolarisation_mv(same_side, r)
