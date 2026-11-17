"""Reproduce every result: python run_all.py

Runs calibration (seeds A), the main experiment, all 964 rewirings, the
tethered real-fly tests and the sensitivity analysis (seeds B), the v2
random-wiring experiment (its own seed streams), then the figures. About an hour with 10 worker processes.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ("01_calibrate", "02_main", "03_rewiring", "04_fly_benchmarks",
           "05_sensitivity", "08_random_circuits", "09_v2_explore",
           "10_posthoc_v2_mechanism",
           "00_checkpoints", "06_figures")

if __name__ == "__main__":
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for name in SCRIPTS:
        print(f"=== {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "experiments" / f"{name}.py")],
                       check=True, cwd=ROOT, env=env)
