"""Fixed model values. Nothing here is tuned.

Values found by calibration live in minifly/frozen.py, which
experiments/01_calibrate.py writes.
"""
from math import exp, pi
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FROZEN_PATH = Path(__file__).resolve().parent / "frozen.py"

# --- Neurons ---------------------------------------------------------------
# LIF parameters, verbatim from Shiu et al. 2024, as in escape-circuit/src/config.py.
LIF = {
    "v_0": -52.0,     # mV, resting potential
    "v_rst": -52.0,   # mV, reset after spike
    "v_th": -45.0,    # mV, spike threshold
    "t_mbr": 20.0,    # ms, membrane time constant
    "tau": 5.0,       # ms, synaptic time constant
    "t_rfc": 2.2,     # ms, refractory period
    "t_dly": 1.8,     # ms, synaptic delay
    "w_syn": 0.275,   # mV, weight per synapse
}
DT_MS = 0.1
DT_S = DT_MS / 1000.0
NOISE_SD_MV = 2.0   # stationary SD of the free (no-threshold) membrane

NEURONS = ("ORN_L", "ORN_R", "PN_L", "PN_R", "IN_L", "IN_R", "MN_L", "MN_R")
ORN_L, ORN_R, PN_L, PN_R, IN_L, IN_R, MN_L, MN_R = range(8)
N_NEURONS = len(NEURONS)

# --- Odour field -----------------------------------------------------------
ODOR_SIGMA_MM = 20.0

# --- Body ------------------------------------------------------------------
BODY_LENGTH_MM = 2.5
HEAD_OFFSET_MM = 1.25          # antennae sit this far ahead of the body centre
ANTENNA_SEP_MM = 2.5           # cartoon scale; real flies: a fraction of a mm
TRACK_WIDTH_MM = 1.0           # distance between the left and right "legs"
MOTOR_TAU_MS = 100.0           # MN spike trains are smoothed over ~one stride
BASE_MN_RATE_HZ = 40.0
BASE_SPEED_MM_S = 10.0
SPEED_PER_HZ = BASE_SPEED_MM_S / BASE_MN_RATE_HZ   # mm/s per Hz of MN rate
ORN_RATE_AT_K_HZ = 50.0

# --- Trials ----------------------------------------------------------------
START_RADIUS_MM = 30.0
FOOD_RADIUS_MM = 5.0
LOST_RADIUS_MM = 60.0
T_MAX_S = 60.0
MAX_STEPS = int(round(T_MAX_S * 1000.0 / DT_MS))

# Half-saturation of odour transduction: the concentration at the start radius.
ODOR_K = exp(-START_RADIUS_MM ** 2 / (2.0 * ODOR_SIGMA_MM ** 2))

# --- Calibration ------------------------------------------------------------
N_SYN_GRID = (20, 40, 80, 160, 320, 640)
CALIBRATION_SUCCESS_TARGET = 0.80
FALLBACK_ANTENNA_SEP_MM = 5.0

# --- Experiments (sections 6-9) --------------------------------------------
N_TRIALS = 200
CHUNK = 100                     # noise is drawn per chunk of 100 trials
N_TRIALS_REWIRING = 100
THRESHOLD_DOSES_MV = (1.0, 2.0, 3.0, 4.0, 6.0)
S1_ANTENNA_SEPS_MM = (0.3, 0.6, 1.25, 2.5, 5.0)
S2_NOISE_SDS_MV = (1.0, 2.0, 4.0)

# --- Tethered protocols (sections 8 and 14) --------------------------------
TETHER_PRE_MS = 300.0
G1_STIM_MS = 500.0
G3_PULSE_MS = 50.0
G3_WINDOW_MS = 300.0
G3_ASYMMETRY = 0.05
STEER_STIM_MS = 1000.0

# --- Metrics ---------------------------------------------------------------
TRAJ_SAMPLE_STEPS = 100         # store the trajectory every 10 ms
CIRCLING_WINDOW_S = 10.0
CIRCLING_MAX_DISPLACEMENT_MM = BODY_LENGTH_MM
CIRCLING_MIN_TURN_RAD = 2.0 * pi

# --- Statistics ------------------------------------------------------------
ALPHA = 0.05
N_BOOT = 10_000

# --- Seeds -----------------------------------------------------------------
SEED_BASE = 20261103
STREAM_CALIBRATION = 1          # seeds A
STREAM_MAIN = 2                 # seeds B
