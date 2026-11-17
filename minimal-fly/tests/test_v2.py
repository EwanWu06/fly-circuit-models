"""Second version: random wiring rules, layout and the uniform-odour control."""
import numpy as np

from minifly import config as cfg, inference as inf, sim, v2

DRIVES = sim.Drives(mn_drive=2.5, orn_gain=6.0)     # test values


def test_random_circuits_follow_the_registered_rules():
    total = 0
    for k in range(200):
        edges, signs = v2.random_circuit(k)
        assert edges == v2.random_circuit(k)[0]                 # reproducible
        for pre, post, s in edges:
            assert pre != post
            assert pre not in v2.MNS
            assert not (pre in v2.ORNS and post in v2.MNS)       # no ORN -> MN
            assert post not in v2.ORNS
            if pre in v2.ORNS:
                assert s == 1
            else:
                assert s == signs[v2.INTS.index(pre)]
        total += len(edges)
    mean = total / 200
    assert abs(mean - 0.5 * (56 + 12 + 8)) < 2.0                # about 38 edges


def test_designed_circuit_on_the_v2_layout():
    edges, signs = v2.designed_circuit()
    assert len(edges) == 32
    assert v2.crossed_index(edges, signs) == 14
    w = v2.weight_matrix(edges, 320)
    assert np.isclose(w[0, 14], 320 / 7 * cfg.LIF["w_syn"])
    assert np.isclose(w[17, 14], -320 * cfg.LIF["w_syn"])


def test_crossed_index_under_mirror_and_leg_swap():
    """A full left-right mirror keeps an approacher an approacher (X kept);
    swapping only the legs reverses steering (X negated)."""
    sides = {**{l: r for l, r in zip(v2.ORN_LEFT, v2.ORN_RIGHT)},
             **{r: l for l, r in zip(v2.ORN_LEFT, v2.ORN_RIGHT)}}
    legs = {v2.MN_L: v2.MN_R, v2.MN_R: v2.MN_L}
    full = {**sides, **legs}
    for k in range(50):
        edges, signs = v2.random_circuit(k)
        x = v2.crossed_index(edges, signs)
        mirrored = [(full.get(p, p), full.get(q, q), s) for p, q, s in edges]
        swapped = [(p, legs.get(q, q), s) for p, q, s in edges]
        assert v2.crossed_index(mirrored, signs) == x
        assert v2.crossed_index(swapped, signs) == -x


def test_uniform_odour_removes_all_position_information():
    x = np.array([30.0, -12.0, 3.0])
    y = np.array([0.0, 20.0, -40.0])
    th = np.array([0.3, 2.0, -1.0])
    sep = np.full(3, 5.0)
    left, right = sim.orn_drives(x, y, np.cos(th), np.sin(th), sep,
                                 np.ones(3, bool), 6.0)
    assert np.allclose(left, 3.0) and np.allclose(right, 3.0)   # gain * K / (K + K)
    left, right = sim.orn_drives(x, y, np.cos(th), np.sin(th), sep,
                                 np.zeros(3, bool), 6.0)
    assert not np.allclose(left, right)


def test_designed_v2_circuit_turns_towards_the_stronger_side():
    edges, _ = v2.designed_circuit()
    c = sim.Condition("D20", v2.weight_matrix(edges, 320), v2.thresholds(), 5.0)
    a = DRIVES.orn_gain
    levels = [[[a / 2, a / 2], [0.6 * a, 0.4 * a]], [[a / 2, a / 2], [0.4 * a, 0.6 * a]]]
    res = sim.run_tethered([c, c], levels, [100.0, 300.0], (0.0, 300.0), range(20),
                           v2.STREAM_TRIALS + 100, DRIVES, layout=v2.LAYOUT)
    assert res.mean_omega[0].mean() > 0 > res.mean_omega[1].mean()


def test_v2_closed_loop_runs_and_is_batch_invariant():
    edges, _ = v2.designed_circuit()
    w = v2.weight_matrix(edges, 320)
    c = sim.Condition("D20", w, v2.thresholds(), 5.0)
    null = sim.Condition("D20-null", w, v2.thresholds(), 5.0, uniform_odor=True)
    kw = dict(max_steps=2000, layout=v2.LAYOUT)
    a = sim.run_closed_loop([c], [0, 1], v2.STREAM_TRIALS + 100, DRIVES, **kw)
    b = sim.run_closed_loop([null, c], [0, 1, 2], v2.STREAM_TRIALS + 100, DRIVES, **kw)
    assert np.array_equal(a.final[0], b.final[1, :2])
    assert not np.array_equal(b.final[0], b.final[1])


def test_two_sided_tests_and_spearman():
    assert inf.binomial_two_sided(5, 10) == 1.0
    assert np.isclose(inf.binomial_two_sided(0, 5), 2 / 32)
    p, n10, n01 = inf.mcnemar_two_sided([1, 1, 1, 0], [0, 0, 0, 0])
    assert (n10, n01) == (3, 0) and np.isclose(p, 0.25)
    rho, p = inf.spearman_greater([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], n_perm=2000)
    assert rho > 0.8 and p < 0.05
    assert np.allclose(inf._ranks([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
