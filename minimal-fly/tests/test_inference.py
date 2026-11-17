import numpy as np

from minifly import inference as inf


def test_wilson_interval_for_4_of_5():
    lo, hi = inf.wilson(4, 5)
    assert round(lo, 2) == 0.38 and round(hi, 2) == 0.96


def test_mcnemar_counts_only_discordant_pairs():
    a = np.array([1, 1, 1, 1, 0, 1, 0], bool)
    b = np.array([0, 0, 0, 1, 0, 1, 0], bool)
    assert np.isclose(inf.mcnemar_greater(a, b), 1 / 8)    # 3 vs 0 discordant
    assert inf.mcnemar_greater(b, a) == 1.0
    assert inf.mcnemar_greater(a, a) == 1.0


def test_cochran_armitage_detects_a_falling_trend():
    doses = [0, 1, 2, 3]
    assert inf.cochran_armitage_decreasing(doses, [90, 70, 50, 30], [100] * 4) < 1e-6
    assert inf.cochran_armitage_decreasing(doses, [30, 50, 70, 90], [100] * 4) > 0.99
    assert inf.cochran_armitage_decreasing(doses, [0, 0, 0, 0], [100] * 4) == 1.0


def test_holm_step_down():
    adj, rej = inf.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(rej) == [True, False, False, True]


def test_bh_prefix_rule_as_in_fly():
    # Ranks 2 and 3 miss their own thresholds, but rank 4 passes, so all four
    # are rejected (the step-up prefix rule).
    p = np.array([0.01, 0.04, 0.03, 0.045])
    assert list(inf.bh(p, 0.05)) == [True, True, True, True]
    assert list(inf.bh(np.array([0.01, 0.04, 0.03, 0.20]), 0.05)) == \
        [True, False, False, False]


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    v = np.random.default_rng(1).normal(1.0, 1.0, 200)
    ci = inf.bootstrap_mean_ci(v, seed=5)
    assert ci == inf.bootstrap_mean_ci(v, seed=5)
    assert ci[0] < v.mean() < ci[1]


def test_z_test():
    assert inf.z_test_two_sided(np.array([1.0, -1.0, 1.0, -1.0])) == 1.0
    assert inf.z_test_two_sided(np.full(10, 2.0)) == 0.0
