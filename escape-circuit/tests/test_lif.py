import numpy as np
import pytest

from src import config, lif
from src.circuit import Circuit


def _single_neuron() -> Circuit:
    return Circuit(
        body_ids=np.array([1], dtype=np.int64),
        types=np.array(["TEST"], dtype="<U16"),
        sides=np.array(["L"], dtype="<U1"),
        pre=np.empty(0, dtype=np.int32),
        post=np.empty(0, dtype=np.int32),
        weight=np.empty(0, dtype=np.float64),
    )


def test_no_input_stays_at_rest():
    c = _single_neuron()
    t = np.arange(0.0, 50.0, config.DT_MS)
    r = lif.simulate(c, np.zeros((1, t.size)), t)
    assert np.allclose(r.v, config.LIF["v_0"])
    assert not r.spikes.any()


def test_matches_fine_step_charging_curve():
    """With constant drive and no spiking, v(t) matches a fine-step reference.

    dg/dt = -g/tau + I
    dv/dt = (v_0 - v + g)/t_mbr
    The reference integrates these by forward Euler at dt = 0.001 ms. It is
    not an analytic solution. The engine at dt = 0.1 ms must stay within
    0.05 mV of it.
    """
    c = _single_neuron()
    t = np.arange(0.0, 30.0, config.DT_MS)
    drive = np.full((1, t.size), 0.05)          # small enough not to spike
    got = lif.simulate(c, drive, t).v[0]

    p, fine_dt = config.LIF, 0.001
    n = int(round(30.0 / fine_dt))
    v, g = p["v_0"], 0.0
    ref = []
    for i in range(n):
        ref.append(v)
        g += fine_dt * (-g / p["tau"] + 0.05)
        v += fine_dt * ((p["v_0"] - v + g) / p["t_mbr"])
    ref = np.array(ref)[:: int(round(config.DT_MS / fine_dt))][: t.size]
    assert np.max(np.abs(got - ref)) < 0.05    # mV


def test_spike_resets_and_respects_refractory():
    c = _single_neuron()
    t = np.arange(0.0, 100.0, config.DT_MS)
    r = lif.simulate(c, np.full((1, t.size), 5.0), t)
    idx = np.flatnonzero(r.spikes[0])
    assert idx.size >= 2
    gaps_ms = np.diff(idx) * config.DT_MS
    assert gaps_ms.min() >= config.LIF["t_rfc"]
    assert r.v[0, idx[0] + 1] == pytest.approx(config.LIF["v_rst"], abs=1e-9)


def _converging_circuit(n_pre: int) -> Circuit:
    """n_pre identical presynaptic neurons, each with one excitatory edge
    (weight 1.0) onto a single shared target neuron (the last index)."""
    n = n_pre + 1
    return Circuit(
        body_ids=np.arange(1, n + 1, dtype=np.int64),
        types=np.array(["PRE"] * n_pre + ["POST"], dtype="<U16"),
        sides=np.array(["L"] * n, dtype="<U1"),
        pre=np.arange(n_pre, dtype=np.int32),
        post=np.full(n_pre, n_pre, dtype=np.int32),
        weight=np.full(n_pre, 1.0),
    )


def test_converging_simultaneous_spikes_all_reach_target():
    """Several presynaptic neurons can spike in the same step and converge on
    the same postsynaptic neuron (311 visual neurons -> 2 giant fibers in the
    real circuit). Every converging contribution must be delivered -- a
    buffered `pending[slot][post] += w` would silently drop all but one.
    """
    t = np.arange(0.0, 100.0, config.DT_MS)

    def peak_depolarisation_mv(n_pre: int) -> tuple[float, bool]:
        c = _converging_circuit(n_pre)
        drive = np.zeros((c.n_neurons, t.size))
        drive[:n_pre] = 5.0             # identical drive -> identical, simultaneous spikes
        r = lif.simulate(c, drive, t)
        target = n_pre                  # the single POST neuron's index
        peak = float(r.v[target].max() - config.LIF["v_0"])
        return peak, bool(r.spikes[target].any())

    one, spiked_one = peak_depolarisation_mv(1)
    two, spiked_two = peak_depolarisation_mv(2)

    assert not spiked_one and not spiked_two, "target must stay subthreshold"
    assert two == pytest.approx(2 * one, rel=1e-9)


def test_inhibitory_edge_lowers_postsynaptic_potential():
    c = Circuit(
        body_ids=np.array([1, 2], dtype=np.int64),
        types=np.array(["PRE", "POST"], dtype="<U16"),
        sides=np.array(["L", "L"], dtype="<U1"),
        pre=np.array([0], dtype=np.int32),
        post=np.array([1], dtype=np.int32),
        weight=np.array([-50.0]),
    )
    t = np.arange(0.0, 100.0, config.DT_MS)
    drive = np.zeros((2, t.size))
    drive[0] = 5.0
    r = lif.simulate(c, drive, t)
    assert r.v[1].min() < config.LIF["v_0"] - 0.1
