"""Project-wide constants. Nothing here is tuned."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0"
    "/connectome-data/flat-connectome"
)
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}
EXPECTED_BYTES = {
    "annotations": 14_483_314,
    "weights": 1_051_241_946,
}
# SHA-256 of the exact files every result in this repository was computed
# from. src/download.py checks a freshly downloaded file against these before
# keeping it; files already present are not re-hashed.
EXPECTED_SHA256 = {
    "annotations":
        "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "weights":
        "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    "neurotransmitters":
        "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
}

# LIF parameters, verbatim from Shiu et al. 2024 model.py. Do not tune.
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

# Stimulus protocol (Ache et al. 2019)
RV_VALUES_MS = (10.0, 20.0, 40.0, 80.0)
THETA_START_DEG = 10.0
THETA_END_DEG = 63.0

# Converts the dimensionless stimulus drive to mV/ms. Fixed at 8.0 under a
# rule that reads the input layer only: the smallest gain, in steps of 1.0,
# at which the fastest condition (r/v = 10 ms) recruits the entire 311-neuron
# visual input layer. Recorded: 3.0 -> 186, 5.0 -> 293, 6.0 -> 304, 8.0 -> 311.
# Two qualifications: a post hoc re-check (experiments/05_posthoc_gain_rule.py)
# finds that the rule gives 7.0 at sigma = 15 deg, where these counts
# reproduce, and 6.0 at the final sigma of 5 deg; and some B1 outcomes had
# been glimpsed in an unrecorded gain x sigma scan before the value was fixed.
INPUT_GAIN_MV_PER_MS = 8.0

# Per-neuron temporal jitter applied to the visual input layer, in ms.
# Set to t_mbr / 2: the membrane time constant is the window over which the
# giant fiber integrates, so this is the scale at which asynchrony matters.
# Derived from a literature-fixed constant, never from a benchmark outcome.
INPUT_JITTER_MS = LIF["t_mbr"] / 2.0

MIN_SYNAPSE_WEIGHT = 5
SIGN_BY_NT = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
