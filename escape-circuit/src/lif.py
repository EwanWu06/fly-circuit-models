"""Leaky integrate-and-fire engine.

Parameters and equations follow Shiu et al. 2024 (Nature):

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau

A spike (v > v_th) resets v to v_rst, zeroes g, and holds the neuron
refractory for t_rfc. Presynaptic spikes add w to the postsynaptic g after a
delay of t_dly, where w = signed_synapse_count * w_syn.

Implemented directly in NumPy rather than Brian2: the circuit is small, and
every line needs to be inspectable.
"""
from dataclasses import dataclass

import numpy as np

from . import config
from .circuit import Circuit


@dataclass(frozen=True)
class SimResult:
    v: np.ndarray        # (N, T) mV
    spikes: np.ndarray   # (N, T) bool
    t_ms: np.ndarray     # (T,) ms


def simulate(c: Circuit, drive: np.ndarray, t_ms: np.ndarray,
             gain_mv_per_ms: float = 1.0) -> SimResult:
    p = config.LIF
    dt = float(t_ms[1] - t_ms[0])
    n, steps = c.n_neurons, t_ms.size

    delay_steps = max(1, int(round(p["t_dly"] / dt)))
    refractory_steps = int(round(p["t_rfc"] / dt))
    syn_w = c.weight * p["w_syn"]                 # mV per presynaptic spike

    v = np.full(n, p["v_0"], dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    locked_until = np.zeros(n, dtype=np.int64)
    pending = np.zeros((delay_steps + 1, n), dtype=np.float64)

    v_out = np.empty((n, steps), dtype=np.float64)
    spikes = np.zeros((n, steps), dtype=bool)

    for step in range(steps):
        v_out[:, step] = v
        free = step >= locked_until

        g += pending[step % pending.shape[0]]
        pending[step % pending.shape[0]] = 0.0

        g += dt * (-g / p["tau"] + drive[:, step] * gain_mv_per_ms)
        v[free] += dt * ((p["v_0"] - v[free] + g[free]) / p["t_mbr"])

        fired = free & (v > p["v_th"])
        if fired.any():
            spikes[fired, step] = True
            v[fired] = p["v_rst"]
            g[fired] = 0.0
            locked_until[fired] = step + refractory_steps
            active = fired[c.pre]
            if active.any():
                slot = (step + delay_steps) % pending.shape[0]
                np.add.at(pending[slot], c.post[active], syn_w[active])

    return SimResult(v=v_out, spikes=spikes, t_ms=t_ms)
