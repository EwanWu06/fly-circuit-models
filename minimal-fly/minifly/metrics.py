"""Trial metrics M1-M5."""
import numpy as np

from . import config as cfg


def trial_metrics(res, c: int) -> dict:
    """Per-trial arrays for condition index c of a ClosedLoopResult."""
    status = res.status[c]
    steps = res.end_step[c].astype(float)
    time_s = steps * cfg.DT_S
    path = res.path_mm[c]
    success = status == 1
    efficiency = np.full(status.size, np.nan)
    efficiency[success] = ((cfg.START_RADIUS_MM - cfg.FOOD_RADIUS_MM)
                           / path[success])
    return {
        "success": success,
        "lost": status == 2,
        "timeout": status == 3,
        "time_s": time_s,
        "path_mm": path,
        "efficiency": efficiency,
        "approach": res.cos_sum[c] / steps,        # M4
        "speed": path / time_s,                     # M5
    }


def is_circling(traj: np.ndarray) -> bool:
    """Circling in place: within
    some 10-s window, net displacement under one body length while the
    signed heading change is at least 360 degrees. traj: (samples, 3)
    x, y, unwrapped theta every 10 ms, NaN after the trial ended."""
    valid = int(np.sum(~np.isnan(traj[:, 0])))
    w = int(round(cfg.CIRCLING_WINDOW_S * 1000 / (cfg.TRAJ_SAMPLE_STEPS * cfg.DT_MS)))
    if valid <= w:
        return False
    starts = np.arange(0, valid - w, 10)          # slide by 100 ms
    disp = np.hypot(traj[starts + w, 0] - traj[starts, 0],
                    traj[starts + w, 1] - traj[starts, 1])
    turn = np.abs(traj[starts + w, 2] - traj[starts, 2])
    return bool(np.any((disp < cfg.CIRCLING_MAX_DISPLACEMENT_MM)
                       & (turn >= cfg.CIRCLING_MIN_TURN_RAD)))
