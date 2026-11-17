"""Chapter 7, checkpoint 3: trace single spikes through the wiring by hand."""
import numpy as np

from minifly import circuit, config as cfg
from minifly.neuron import DELAY_STEPS, Network

N_SYN = 40
W_EDGE = N_SYN * cfg.LIF["w_syn"]


def _one_spike_from(neuron):
    """Fire `neuron` once at step 0 and return the input queued for later."""
    w = circuit.weight_matrix(circuit.edges_of(circuit.BASELINE), N_SYN)
    net = Network([w], [circuit.thresholds()], [0.0], [0])
    net.v[0, neuron] = cfg.LIF["v_th"] + 1.0
    zero = np.zeros((1, cfg.N_NEURONS))
    fired = net.step(0, zero)
    assert fired[0, neuron] and fired.sum() == 1
    return net, net.pending[DELAY_STEPS % net.pending.shape[0], 0].copy()


def test_baseline_has_the_eight_registered_edges():
    edges = circuit.BASELINE
    assert len(edges) == 8
    assert sum(1 for _, _, s in edges.values() if s < 0) == 2
    assert edges["e7"] == (cfg.PN_L, cfg.MN_R, 1)
    assert edges["e8"] == (cfg.PN_R, cfg.MN_L, 1)


def test_orn_spike_reaches_its_pn_and_in_after_1_8_ms():
    net, queued = _one_spike_from(cfg.ORN_L)
    expected = np.zeros(cfg.N_NEURONS)
    expected[[cfg.PN_L, cfg.IN_L]] = W_EDGE
    assert np.array_equal(queued, expected)
    zero = np.zeros((1, cfg.N_NEURONS))
    for s in range(1, DELAY_STEPS):
        net.step(s, zero)
        assert net.g[0, cfg.PN_L] == 0.0
    net.step(DELAY_STEPS, zero)                       # 18 steps = 1.8 ms
    assert net.g[0, cfg.PN_L] > 0 and net.g[0, cfg.IN_L] > 0


def test_in_spike_inhibits_the_opposite_pn():
    _, queued = _one_spike_from(cfg.IN_L)
    expected = np.zeros(cfg.N_NEURONS)
    expected[cfg.PN_R] = -W_EDGE
    assert np.array_equal(queued, expected)


def test_pn_spike_excites_the_opposite_mn():
    _, queued = _one_spike_from(cfg.PN_L)
    expected = np.zeros(cfg.N_NEURONS)
    expected[cfg.MN_R] = W_EDGE
    assert np.array_equal(queued, expected)


def test_perturbations():
    flipped = circuit.flip_signs(circuit.BASELINE, circuit.CROSSED_MOTOR_EDGES)
    assert flipped["e7"][2] == -1 and flipped["e8"][2] == -1 and flipped["e1"][2] == 1
    assert set(circuit.drop(circuit.BASELINE, circuit.COMPARISON_EDGES)) == \
        {"e1", "e2", "e3", "e4", "e7", "e8"}
    v_th = circuit.thresholds(2.0, circuit.SENSORY_AND_INTER)
    assert np.all(v_th[:6] == cfg.LIF["v_th"] + 2.0)
    assert np.all(v_th[6:] == cfg.LIF["v_th"])
