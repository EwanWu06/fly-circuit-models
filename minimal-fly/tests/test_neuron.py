"""Chapter 7, checkpoint 1: the LIF neuron (integrate, fire, reset)."""
import numpy as np

from minifly import circuit, config as cfg
from minifly.neuron import DELAY_STEPS, REFRACTORY_STEPS, Network, noise_sigma_for_sd
from tests import reference_fly_lif as fly


def test_matches_fly_engine_step_for_step_without_noise():
    """Same equations and update order as escape-circuit/src/lif.py: identical numbers.
    200 synapses per edge, so that spikes propagate through every layer."""
    n_syn = 200
    edges = sorted(circuit.edges_of(circuit.BASELINE))      # pre-sorted, as the reference add.at order
    ref = fly.Circuit(pre=np.array([e[0] for e in edges]),
                      post=np.array([e[1] for e in edges]),
                      weight=np.array([float(e[2] * n_syn) for e in edges]),
                      n=cfg.N_NEURONS)
    steps = 6000
    t_ms = np.arange(steps) * cfg.DT_MS
    rng = np.random.default_rng(7)
    drive = np.zeros((cfg.N_NEURONS, steps))
    drive[[cfg.ORN_L, cfg.ORN_R, cfg.MN_L, cfg.MN_R]] = rng.uniform(0.0, 4.0, (4, steps))
    expected = fly.simulate(ref, drive, t_ms, gain_mv_per_ms=1.0)

    net = Network([circuit.weight_matrix(edges, n_syn)], [circuit.thresholds()],
                  [0.0], [0])
    v_out = np.empty((cfg.N_NEURONS, steps))
    spikes = np.zeros((cfg.N_NEURONS, steps), dtype=bool)
    for s in range(steps):
        v_out[:, s] = net.v[0]
        spikes[:, s] = net.step(s, drive[:, s][None, :])[0]

    assert expected.spikes[[cfg.PN_L, cfg.PN_R, cfg.IN_L, cfg.IN_R]].any()
    assert np.array_equal(v_out, expected.v)
    assert np.array_equal(spikes, expected.spikes)


def test_integrate_fire_reset_and_refractory():
    net = Network(np.zeros((1, 1, 1)), [[cfg.LIF["v_th"]]], [0.0], [0])
    drive = np.full((1, 1), 3.0)
    v, fired_at = [], []
    for s in range(3000):
        v.append(net.v[0, 0])
        if net.step(s, drive)[0, 0]:
            fired_at.append(s)
            assert net.v[0, 0] == cfg.LIF["v_rst"]
            assert net.g[0, 0] == 0.0
    v = np.array(v)
    assert len(fired_at) >= 5
    assert np.all(np.diff(v[: fired_at[0]]) > 0)          # integrates upward
    assert np.all(np.diff(fired_at) > REFRACTORY_STEPS)   # refractory respected
    assert np.ptp(np.diff(fired_at[1:])) <= 1             # regular without noise


def test_noise_gives_registered_free_membrane_sd():
    sigma = noise_sigma_for_sd(cfg.NOISE_SD_MV)
    n = 200
    net = Network(np.zeros((1, 1, 1)), [[np.inf]], [sigma], np.zeros(n, int))
    rng = np.random.default_rng(3)
    zero = np.zeros((n, 1))
    samples = []
    for s in range(30_000):
        net.step(s, zero, rng.standard_normal((n, 1)))
        if s >= 2000 and s % 50 == 0:
            samples.append(net.v[:, 0].copy())
    sd = np.std(np.concatenate(samples) - cfg.LIF["v_0"])
    assert abs(sd - cfg.NOISE_SD_MV) / cfg.NOISE_SD_MV < 0.03


def test_delay_constants_match_fly():
    assert DELAY_STEPS == 18 and REFRACTORY_STEPS == 22
