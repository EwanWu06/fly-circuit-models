"""Leaky integrate-and-fire engine, batched over many independent copies.

Equations, parameters and update order are those of escape-circuit/src/lif.py, which
follows Shiu et al. 2024:

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau + drive

A spike (v > v_th) resets v to v_rst, zeroes g, and holds the neuron
refractory for t_rfc. Its signed weight reaches each target's g after t_dly.
tests/test_neuron.py checks step-for-step identity with that engine when the
noise is off.

The only addition is noise: independent white noise added to g, so g is an
Ornstein-Uhlenbeck process.
"""
import numpy as np

from . import config

P = config.LIF
DELAY_STEPS = max(1, int(round(P["t_dly"] / config.DT_MS)))
REFRACTORY_STEPS = int(round(P["t_rfc"] / config.DT_MS))


def noise_sigma_for_sd(sd_mv: float, dt: float = config.DT_MS) -> float:
    """Noise amplitude (mV per sqrt(ms)) that gives the free membrane a
    stationary SD of `sd_mv`.

    "Free" means no threshold. The variance is computed exactly for the
    discrete Euler update used in Network.step, by solving the Lyapunov
    equation P = F P F' + Q for the deviations (g, v) from rest.
    """
    a = 1.0 - dt / P["tau"]
    b = dt / P["t_mbr"]
    f = np.array([[a, 0.0], [b * a, 1.0 - b]])
    gain = np.sqrt(dt) * np.array([1.0, b])
    q = np.outer(gain, gain)
    vec_p = np.linalg.solve(np.eye(4) - np.kron(f, f), q.reshape(-1))
    var_v_unit = vec_p.reshape(2, 2)[1, 1]
    return float(sd_mv / np.sqrt(var_v_unit))


class Network:
    """B independent copies of an N-neuron LIF network.

    Copies can belong to different conditions: copy i uses the weight
    matrix `weights_by_cond[cond_idx[i]]` ([pre, post], mV added to the
    target's g per presynaptic spike), its own thresholds and its own
    noise amplitude.
    """

    def __init__(self, weights_by_cond, v_th_by_cond, sigma_by_cond, cond_idx):
        self.w = np.asarray(weights_by_cond, dtype=np.float64)
        self.cond = np.asarray(cond_idx, dtype=np.int64)
        n_copies, n = self.cond.size, self.w.shape[1]
        self.n = n
        self.v_th = np.asarray(v_th_by_cond, dtype=np.float64)[self.cond]
        self.sigma = np.asarray(sigma_by_cond, dtype=np.float64)[self.cond]
        self.v = np.full((n_copies, n), P["v_0"], dtype=np.float64)
        self.g = np.zeros((n_copies, n), dtype=np.float64)
        self.locked_until = np.zeros((n_copies, n), dtype=np.int64)
        self.pending = np.zeros((DELAY_STEPS + 1, n_copies, n), dtype=np.float64)
        self.sqrt_dt = np.sqrt(config.DT_MS)

    def step(self, step: int, drive: np.ndarray, xi=None) -> np.ndarray:
        """Advance one time step.

        drive: (B, N) external input in mV/ms. xi: (B, N) standard normal
        draws, or None for no noise. Returns the (B, N) boolean spike array.
        """
        dt = config.DT_MS
        ring = self.pending.shape[0]
        slot = step % ring
        free = step >= self.locked_until

        self.g += self.pending[slot]
        self.pending[slot] = 0.0

        self.g += dt * (-self.g / P["tau"] + drive)
        if xi is not None:
            self.g += (self.sigma * self.sqrt_dt)[:, None] * xi
        v_next = self.v + dt * ((P["v_0"] - self.v + self.g) / P["t_mbr"])
        self.v = np.where(free, v_next, self.v)

        fired = free & (self.v > self.v_th)
        if fired.any():
            self.v[fired] = P["v_rst"]
            self.g[fired] = 0.0
            self.locked_until[fired] = step + REFRACTORY_STEPS
            out = self.pending[(step + DELAY_STEPS) % ring]
            # Presynaptic neurons in index order, so that when several spikes
            # reach one target in the same step they are summed in the same
            # order as escape-circuit's np.add.at over pre-sorted edges.
            for j in np.flatnonzero(fired.any(axis=0)):
                rows = np.flatnonzero(fired[:, j])
                out[rows] += self.w[self.cond[rows], j]
        return fired

    def keep(self, mask: np.ndarray) -> None:
        """Drop the copies where mask is False."""
        self.cond = self.cond[mask]
        self.v_th = self.v_th[mask]
        self.sigma = self.sigma[mask]
        self.v = self.v[mask]
        self.g = self.g[mask]
        self.locked_until = self.locked_until[mask]
        self.pending = self.pending[:, mask]
