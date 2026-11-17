"""Reproduce every result from scratch. Expect ~30 minutes on first run."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "experiments/00_validate_extraction.py",
    "experiments/01_benchmarks.py",
    "experiments/02_perturbation.py",
    # POST HOC diagnostics, not part of the pre-registration (see their
    # docstrings).
    "experiments/04_posthoc_gf_synchrony.py",
    "experiments/05_posthoc_gain_rule.py",   # POST HOC, likewise
]

if __name__ == "__main__":
    # A script's own directory starts sys.path, not the repo root, so
    # `from src import ...` needs the root on PYTHONPATH.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, step], check=True, cwd=ROOT, env=env)
    print("\nDone. See results/.")
