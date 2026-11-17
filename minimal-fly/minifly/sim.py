"""Closed-loop and tethered simulations, batched over conditions and trials.

Every random number a trial uses depends only on (stream, trial id): its
start comes from `start_states`, and its noise from a generator shared by the
100 trials of its chunk. Conditions, batch composition and worker splits
therefore never change a trial's inputs, so any condition can be re-run alone
and give the same numbers (tests/test_sim.py checks this).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np

from . import body, environment
from . import config as cfg
from .neuron import Network, noise_sigma_for_sd


@dataclass(frozen=True)
class Condition:
    name: str
    weights: np.ndarray            # (N, N) [pre, post], mV per spike
    v_th: np.ndarray               # (N,) mV
    antenna_sep_mm: float = cfg.ANTENNA_SEP_MM
    noise_sd_mv: float = cfg.NOISE_SD_MV
    uniform_odor: bool = False     # v2 control: every antenna smells C = K everywhere


@dataclass(frozen=True)
class Layout:
    """Which neurons receive each antenna's odour and which drive the legs."""
    n: int
    orn_left: tuple
    orn_right: tuple
    mn_left: int
    mn_right: int


V1 = Layout(cfg.N_NEURONS, (cfg.ORN_L,), (cfg.ORN_R,), cfg.MN_L, cfg.MN_R)


@dataclass(frozen=True)
class Drives:
    mn_drive: float    # mV/ms, tonic input to both MNs
    orn_gain: float    # A: ORN drive = A * C / (C + K), mV/ms


def start_states(stream: int, n: int):
    """Start position (on the start circle) and heading of trials 0..n-1."""
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.SEED_BASE, stream, 999_999]))
    u = rng.uniform(0.0, 2.0 * np.pi, size=(n, 2))
    angle, heading = u[:, 0], u[:, 1]
    r = cfg.START_RADIUS_MM
    return r * np.cos(angle), r * np.sin(angle), heading


class NoiseSource:
    """Standard normal noise for each trial, identical in every run."""

    def __init__(self, stream: int, trial_ids, n: int = cfg.N_NEURONS):
        self.n = n
        trial_ids = np.asarray(trial_ids, dtype=np.int64)
        chunks = np.unique(trial_ids // cfg.CHUNK)
        self.gens = [np.random.default_rng(
            np.random.SeedSequence([cfg.SEED_BASE, stream, int(c)]))
            for c in chunks]
        position = {int(c): k for k, c in enumerate(chunks)}
        self.row = np.array([position[int(t // cfg.CHUNK)] * cfg.CHUNK
                             + int(t % cfg.CHUNK) for t in trial_ids])

    def draw(self) -> np.ndarray:
        return np.concatenate([g.standard_normal((cfg.CHUNK, self.n))
                               for g in self.gens])


def orn_drives(x, y, cos_t, sin_t, sep, uniform, gain):
    """ORN drive at the left and right antenna. Where `uniform` is set, both
    antennae smell the start-radius concentration K wherever the fly is."""
    lx, ly, rx, ry = environment.antenna_positions(x, y, cos_t, sin_t, sep)
    c_left = environment.concentration(lx, ly)
    c_right = environment.concentration(rx, ry)
    if uniform.any():
        c_left = np.where(uniform, cfg.ODOR_K, c_left)
        c_right = np.where(uniform, cfg.ODOR_K, c_right)
    return (environment.transduce(c_left, gain),
            environment.transduce(c_right, gain))


def _network(conditions, cond_of) -> Network:
    return Network([c.weights for c in conditions],
                   [c.v_th for c in conditions],
                   [noise_sigma_for_sd(c.noise_sd_mv) for c in conditions],
                   cond_of)


@dataclass
class ClosedLoopResult:
    names: list
    trial_ids: np.ndarray
    status: np.ndarray       # (C, T) 1 success, 2 lost, 3 timeout
    end_step: np.ndarray     # (C, T) steps until the trial ended
    path_mm: np.ndarray      # (C, T)
    cos_sum: np.ndarray      # (C, T) sum over steps of cos(heading - food dir)
    final: np.ndarray        # (C, T, 3) x, y, theta at the end
    traj: object = None      # (C, T, samples, 3) float32, NaN after the end
    spikes: object = None    # (k, 4) int: cond, trial position, step, neuron

    def merge(self, other: "ClosedLoopResult") -> "ClosedLoopResult":
        cat = lambda a, b: None if a is None else np.concatenate([a, b])
        spikes = None
        if self.spikes is not None:
            shifted = other.spikes.copy()
            shifted[:, 0] += len(self.names)
            spikes = np.concatenate([self.spikes, shifted])
        return ClosedLoopResult(
            self.names + other.names, self.trial_ids,
            *(np.concatenate([getattr(self, f), getattr(other, f)])
              for f in ("status", "end_step", "path_mm", "cos_sum", "final")),
            cat(self.traj, other.traj), spikes)


def run_closed_loop(conditions, trial_ids, stream: int, drives: Drives,
                    record_traj: bool = False, n_spike_trials: int = 0,
                    max_steps: int = cfg.MAX_STEPS,
                    layout: Layout = V1) -> ClosedLoopResult:
    """Free-walking trials in the odour field."""
    n_cond, n_trial = len(conditions), len(trial_ids)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    cond_of = np.repeat(np.arange(n_cond), n_trial)
    tpos_of = np.tile(np.arange(n_trial), n_cond)
    n_all = n_cond * n_trial

    net = _network(conditions, cond_of)
    sep = np.array([c.antenna_sep_mm for c in conditions])[cond_of]
    uniform = np.array([c.uniform_odor for c in conditions])[cond_of]
    left, right = list(layout.orn_left), list(layout.orn_right)
    mn = [layout.mn_left, layout.mn_right]
    sx, sy, sh = start_states(stream, int(trial_ids.max()) + 1)
    x = sx[trial_ids][tpos_of].copy()
    y = sy[trial_ids][tpos_of].copy()
    th = sh[trial_ids][tpos_of].copy()
    rate = np.full((n_all, 2), body.STEADY_RATE_PER_MS)
    drive = np.zeros((n_all, layout.n))
    drive[:, mn] = drives.mn_drive
    noise = NoiseSource(stream, trial_ids, layout.n)
    nrow = noise.row[tpos_of]
    path = np.zeros(n_all)
    cos_sum = np.zeros(n_all)
    gid = np.arange(n_all)
    alive = np.ones(n_all, dtype=bool)

    out_status = np.zeros(n_all, dtype=np.int8)
    out_end = np.zeros(n_all, dtype=np.int64)
    out_path = np.zeros(n_all)
    out_cos = np.zeros(n_all)
    out_final = np.zeros((n_all, 3))
    n_samples = max_steps // cfg.TRAJ_SAMPLE_STEPS + 1
    traj = (np.full((n_all, n_samples, 3), np.nan, dtype=np.float32)
            if record_traj else None)
    spk_mask = tpos_of < n_spike_trials
    spk_rows = np.flatnonzero(spk_mask)
    spk_log = []

    food2, lost2 = cfg.FOOD_RADIUS_MM ** 2, cfg.LOST_RADIUS_MM ** 2

    def finish(rows, code, step):
        g = gid[rows]
        out_status[g] = code
        out_end[g] = step + 1
        out_path[g] = path[rows]
        out_cos[g] = cos_sum[rows]
        out_final[g, 0], out_final[g, 1], out_final[g, 2] = x[rows], y[rows], th[rows]
        alive[rows] = False

    for s in range(max_steps):
        xi = noise.draw()       # drawn every step, so every trial's stream stays aligned
        if not alive.any():
            break
        ct, st = np.cos(th), np.sin(th)
        d_left, d_right = orn_drives(x, y, ct, st, sep, uniform, drives.orn_gain)
        drive[:, left] = d_left[:, None]
        drive[:, right] = d_right[:, None]
        fired = net.step(s, drive, xi[nrow])
        body.smooth_rates(rate, fired[:, mn])
        fwd, omega = body.speeds(rate)

        cos_sum += -(x * ct + y * st) / np.sqrt(x * x + y * y)
        path += fwd * cfg.DT_S
        if traj is not None and s % cfg.TRAJ_SAMPLE_STEPS == 0:
            k = s // cfg.TRAJ_SAMPLE_STEPS
            live = np.flatnonzero(alive)
            traj[gid[live], k, 0] = x[live]
            traj[gid[live], k, 1] = y[live]
            traj[gid[live], k, 2] = th[live]
        if spk_rows.size:
            hit = fired[spk_rows] & alive[spk_rows, None]
            if hit.any():
                r_idx, n_idx = np.nonzero(hit)
                g = gid[spk_rows[r_idx]]
                spk_log.append(np.column_stack(
                    [cond_of[g], tpos_of[g], np.full(g.size, s), n_idx]))

        body.advance(x, y, th, ct, st, fwd, omega)
        d2 = x * x + y * y
        success = alive & (d2 <= food2)
        lost = alive & (d2 >= lost2)
        if success.any():
            finish(np.flatnonzero(success), 1, s)
        if lost.any():
            finish(np.flatnonzero(lost), 2, s)
        if s == max_steps - 1 and alive.any():
            finish(np.flatnonzero(alive), 3, s)

        if s % 1000 == 999 and not alive.all():
            keep = alive.copy()
            net.keep(keep)
            x, y, th, rate, drive = x[keep], y[keep], th[keep], rate[keep], drive[keep]
            sep, nrow, path, cos_sum = sep[keep], nrow[keep], path[keep], cos_sum[keep]
            uniform = uniform[keep]
            gid, spk_mask = gid[keep], spk_mask[keep]
            alive = np.ones(gid.size, dtype=bool)
            spk_rows = np.flatnonzero(spk_mask)

    shape = (n_cond, n_trial)
    return ClosedLoopResult(
        [c.name for c in conditions], trial_ids,
        out_status.reshape(shape), out_end.reshape(shape),
        out_path.reshape(shape), out_cos.reshape(shape),
        out_final.reshape(shape + (3,)),
        None if traj is None else traj.reshape(shape + traj.shape[1:]),
        np.concatenate(spk_log) if spk_log else np.zeros((0, 4), dtype=np.int64))


@dataclass
class TetherResult:
    names: list
    trial_ids: np.ndarray
    mean_omega: np.ndarray   # (C, T) rad/s, mean over the analysis window
    trace: np.ndarray        # (C, bins) trial-mean omega per 1-ms bin
    t0_bin: int              # bin index of time 0 (stimulus onset)

    def merge(self, other: "TetherResult") -> "TetherResult":
        return TetherResult(self.names + other.names, self.trial_ids,
                            np.concatenate([self.mean_omega, other.mean_omega]),
                            np.concatenate([self.trace, other.trace]),
                            self.t0_bin)


def run_tethered(conditions, levels, durations_ms, window_ms, trial_ids,
                 stream: int, drives: Drives, layout: Layout = V1) -> TetherResult:
    """Body fixed in place; the two ORN drives follow a piecewise-constant
    schedule.

    levels: (C, n_segments, 2) ORN drives [left, right] in mV/ms.
    durations_ms: segment lengths, shared by all conditions. Time 0 is the
    end of the first segment. window_ms: (start, end) relative to time 0.
    Turning is read out as the fictive angular velocity of the legs.
    """
    levels = np.asarray(levels, dtype=np.float64)
    n_cond, n_trial = len(conditions), len(trial_ids)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    cond_of = np.repeat(np.arange(n_cond), n_trial)
    tpos_of = np.tile(np.arange(n_trial), n_cond)
    n_all = n_cond * n_trial

    net = _network(conditions, cond_of)
    left, right = list(layout.orn_left), list(layout.orn_right)
    mn = [layout.mn_left, layout.mn_right]
    rate = np.full((n_all, 2), body.STEADY_RATE_PER_MS)
    drive = np.zeros((n_all, layout.n))
    drive[:, mn] = drives.mn_drive
    noise = NoiseSource(stream, trial_ids, layout.n)
    nrow = noise.row[tpos_of]

    seg_steps = [int(round(d / cfg.DT_MS)) for d in durations_ms]
    ends = np.cumsum(seg_steps)
    total, pre = int(ends[-1]), seg_steps[0]
    w0 = pre + int(round(window_ms[0] / cfg.DT_MS))
    w1 = pre + int(round(window_ms[1] / cfg.DT_MS))
    bin_steps = int(round(1.0 / cfg.DT_MS))
    trace = np.zeros((n_cond, total // bin_steps))
    omega_sum = np.zeros(n_all)

    seg = -1
    for s in range(total):
        xi = noise.draw()
        if seg < 0 or s == ends[seg]:
            seg += 1
            drive[:, left] = levels[cond_of, seg, 0][:, None]
            drive[:, right] = levels[cond_of, seg, 1][:, None]
        fired = net.step(s, drive, xi[nrow])
        body.smooth_rates(rate, fired[:, mn])
        _, omega = body.speeds(rate)
        if w0 <= s < w1:
            omega_sum += omega
        trace[:, s // bin_steps] += np.bincount(cond_of, weights=omega,
                                                minlength=n_cond)

    return TetherResult([c.name for c in conditions], trial_ids,
                        (omega_sum / (w1 - w0)).reshape(n_cond, n_trial),
                        trace / (bin_steps * n_trial), pre // bin_steps)


# --- Parallel helpers -------------------------------------------------------

def _pool(n_workers: int) -> ProcessPoolExecutor:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return ProcessPoolExecutor(max_workers=n_workers,
                               mp_context=mp.get_context("spawn"))


def _in_groups(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def parallel_closed_loop(conditions, trial_ids, stream, drives, n_workers=10,
                         group_size=1, log=print, **kw) -> ClosedLoopResult:
    groups = _in_groups(list(conditions), group_size)
    t_start = time.time()
    with _pool(n_workers) as ex:
        futures = {ex.submit(run_closed_loop, g, trial_ids, stream, drives, **kw): i
                   for i, g in enumerate(groups)}
        parts = [None] * len(groups)
        for done, fut in enumerate(as_completed(futures), 1):
            parts[futures[fut]] = fut.result()
            log(f"  closed loop: {done}/{len(groups)} groups done "
                f"({time.time() - t_start:.0f} s)")
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part)
    return result


def parallel_tethered(conditions, levels, durations_ms, window_ms, trial_ids,
                      stream, drives, n_workers=10, group_size=1,
                      log=print, layout: Layout = V1) -> TetherResult:
    idx_groups = _in_groups(list(range(len(conditions))), group_size)
    levels = np.asarray(levels)
    t_start = time.time()
    with _pool(n_workers) as ex:
        futures = {ex.submit(run_tethered, [conditions[i] for i in g], levels[g],
                             durations_ms, window_ms, trial_ids, stream, drives,
                             layout): k
                   for k, g in enumerate(idx_groups)}
        parts = [None] * len(idx_groups)
        for done, fut in enumerate(as_completed(futures), 1):
            parts[futures[fut]] = fut.result()
            log(f"  tethered: {done}/{len(idx_groups)} groups done "
                f"({time.time() - t_start:.0f} s)")
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part)
    return result
