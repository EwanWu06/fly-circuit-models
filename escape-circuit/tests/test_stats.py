import numpy as np

from src import stats


def test_z_score_of_the_null_mean_is_zero():
    null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.z_score(3.0, null) == 0.0


def test_empirical_p_is_never_zero():
    null = np.zeros(100)
    assert stats.empirical_p(999.0, null) == 1 / 101


def test_empirical_p_is_one_for_a_typical_value():
    null = np.array([-1.0, 0.0, 1.0])
    assert stats.empirical_p(0.0, null) == 1.0


def test_bh_fdr_rejects_only_small_p_values():
    p = np.array([0.001, 0.008, 0.4, 0.9])
    assert stats.bh_fdr(p, alpha=0.05).tolist() == [True, True, False, False]


def test_bh_fdr_rejects_nothing_when_all_p_are_large():
    assert not stats.bh_fdr(np.array([0.4, 0.6, 0.9])).any()


def test_bh_fdr_rejects_every_p_up_to_the_largest_passing_rank():
    # Non-contiguous pass pattern [T, F, T] in sorted order requires the prefix rule.
    assert stats.bh_fdr(np.array([0.28, 0.05, 0.25]), alpha=0.3).tolist() == [True, True, True]


def test_z_score_of_a_constant_null_equal_to_observed_is_zero():
    # np.mean of identical copies need not equal the copy exactly due to floating-point precision
    x = 331.90000000007547
    assert stats.z_score(x, np.full(100, x)) == 0.0


def test_z_score_of_a_constant_null_keeps_the_sign_of_the_difference():
    null = np.full(10, 5.0)
    assert stats.z_score(4.0, null) == -np.inf
    assert stats.z_score(6.0, null) == np.inf
