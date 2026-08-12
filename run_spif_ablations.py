"""Sequential launcher for SPIF ablation experiments.

Runs train_spif_af.py → train_spif_bf.py → train_spif_stack.py in order.
Each script runs to completion before the next starts.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_spif_af.py",
    "train_spif_bf.py",
    "train_spif_stack.py",
]

ROOT = Path(__file__).resolve().parent

for i, script in enumerate(SCRIPTS):
    print("=" * 60)
    print(f"[{i+1}/{len(SCRIPTS)}] Starting: {script}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        print("Aborting remaining scripts.")
        sys.exit(result.returncode)
    print(f"\n[{i+1}/{len(SCRIPTS)}] Completed: {script}\n")

print("=" * 60)
print("All SPIF ablation experiments completed.")
print("=" * 60)
