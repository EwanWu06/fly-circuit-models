"""Chapter 7, checkpoint 4: the body walks straight and turns the right way."""
import numpy as np

from minifly import body, config as cfg


def _walk(rate_left_hz, rate_right_hz, seconds=1.0):
    rate = np.array([[rate_left_hz, rate_right_hz]]) / 1000.0
    x, y, th = np.zeros(1), np.zeros(1), np.zeros(1)
    for _ in range(int(seconds / cfg.DT_S)):
        fwd, omega = body.speeds(rate)
        body.advance(x, y, th, np.cos(th), np.sin(th), fwd, omega)
    return x[0], y[0], th[0]


def test_equal_legs_walk_straight_at_base_speed():
    x, y, th = _walk(cfg.BASE_MN_RATE_HZ, cfg.BASE_MN_RATE_HZ)
    assert np.isclose(x, cfg.BASE_SPEED_MM_S) and y == 0.0 and th == 0.0


def test_faster_right_legs_turn_left():
    _, y, th = _walk(35.0, 45.0)
    assert th > 0 and y > 0


def test_steady_spike_train_settles_at_its_rate():
    rate = np.zeros((1, 2))
    period = int(round(1000.0 / cfg.BASE_MN_RATE_HZ / cfg.DT_MS))   # 25 ms
    for s in range(20 * period * 10):
        body.smooth_rates(rate, np.array([[s % period == 0, False]]))
    tail = []
    for s in range(period):
        body.smooth_rates(rate, np.array([[s % period == 0, False]]))
        tail.append(rate[0, 0])
    assert abs(np.mean(tail) * 1000.0 - cfg.BASE_MN_RATE_HZ) < 0.5
