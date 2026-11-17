"""Differential-drive body.

Each side's leg speed is proportional to that side's smoothed MN firing
rate. The right legs moving faster than the left turns the fly left
(counter-clockwise), as for a tank.
"""
import numpy as np

from . import config as cfg

MOTOR_DECAY = float(np.exp(-cfg.DT_MS / cfg.MOTOR_TAU_MS))
MOTOR_KICK = 1.0 / cfg.MOTOR_TAU_MS     # per spike, in spikes/ms
STEADY_RATE_PER_MS = cfg.BASE_MN_RATE_HZ / 1000.0


def smooth_rates(rate_per_ms: np.ndarray, mn_spikes: np.ndarray) -> np.ndarray:
    """Exponential smoothing of the two MN spike trains, in place.

    rate_per_ms: (B, 2) [left, right]; mn_spikes: (B, 2) bool. A steady
    train at f Hz settles at f / 1000 spikes/ms.
    """
    rate_per_ms *= MOTOR_DECAY
    rate_per_ms += mn_spikes * MOTOR_KICK
    return rate_per_ms


def speeds(rate_per_ms: np.ndarray):
    """Forward speed (mm/s) and angular velocity (rad/s, + = left turn)."""
    v_left = cfg.SPEED_PER_HZ * 1000.0 * rate_per_ms[:, 0]
    v_right = cfg.SPEED_PER_HZ * 1000.0 * rate_per_ms[:, 1]
    forward = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / cfg.TRACK_WIDTH_MM
    return forward, omega


def advance(x, y, theta, cos_t, sin_t, forward, omega, dt_s: float = cfg.DT_S):
    """One Euler step of the planar kinematics, in place."""
    x += forward * cos_t * dt_s
    y += forward * sin_t * dt_s
    theta += omega * dt_s
    return x, y, theta
