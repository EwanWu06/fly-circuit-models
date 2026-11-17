"""Gaussian odour field and antenna geometry.

Food sits at the origin. Heading theta is measured counter-clockwise from +x,
so the fly's left is the heading rotated by +90 degrees.
"""
import numpy as np

from . import config as cfg


def concentration(x, y, sigma: float = cfg.ODOR_SIGMA_MM):
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))


def antenna_positions(x, y, cos_t, sin_t, sep, head: float = cfg.HEAD_OFFSET_MM):
    """Left and right antenna positions for a body at (x, y) with heading
    (cos_t, sin_t). Returns (lx, ly, rx, ry)."""
    hx = x + head * cos_t
    hy = y + head * sin_t
    half = 0.5 * sep
    lx = hx - half * sin_t
    ly = hy + half * cos_t
    rx = hx + half * sin_t
    ry = hy - half * cos_t
    return lx, ly, rx, ry


def transduce(c, gain: float, k: float = cfg.ODOR_K):
    """ORN drive in mV/ms: Michaelis-Menten in concentration."""
    return gain * c / (c + k)
