"""Chapter 7, checkpoint 2: the odour field and what each antenna smells."""
import numpy as np

from minifly import config as cfg, environment as env


def _antennae(x, y, theta, sep=cfg.ANTENNA_SEP_MM):
    lx, ly, rx, ry = env.antenna_positions(np.array([x]), np.array([y]),
                                           np.cos([theta]), np.sin([theta]), sep)
    return env.concentration(lx, ly)[0], env.concentration(rx, ry)[0]


def test_concentration_peaks_at_food_and_falls_with_distance():
    r = np.linspace(0, 60, 61)
    c = env.concentration(r, 0 * r)
    assert c[0] == 1.0
    assert np.all(np.diff(c) < 0)
    assert np.isclose(env.concentration(np.array([cfg.START_RADIUS_MM]), np.array([0.0]))[0],
                      cfg.ODOR_K)


def test_left_antenna_is_on_the_left():
    lx, ly, rx, ry = env.antenna_positions(np.array([0.0]), np.array([0.0]),
                                           np.array([1.0]), np.array([0.0]), 2.0)
    assert ly[0] > 0 > ry[0]              # heading +x: left is +y
    assert lx[0] == rx[0] == cfg.HEAD_OFFSET_MM


def test_facing_the_food_both_antennae_smell_the_same():
    left, right = _antennae(30.0, 0.0, np.pi)        # at +x, facing the origin
    assert np.isclose(left, right)


def test_food_on_the_left_left_antenna_smells_more():
    # At (30, 0) facing +y, the fly's left is -x, towards the food.
    left, right = _antennae(30.0, 0.0, np.pi / 2)
    assert left > right
    # Facing -y, its left is +x, away from the food.
    left, right = _antennae(30.0, 0.0, -np.pi / 2)
    assert right > left


def test_transduction_is_half_maximal_at_k():
    assert np.isclose(env.transduce(cfg.ODOR_K, 2.0), 1.0)
