"""Ablation study: measure foreground/background inversion rate.

Runs 20 short training sessions (5 epochs each) by calling train_dual.py:
    1-10: without low-frequency constraint (--lambda_low_freq 0)
    11-20: with low-frequency constraint (--lambda_low_freq 0.5)

Usage:
    python run_ablation.py
"""

import subprocess
import sys
from pathlib import Path


BASE_ARGS = [
    sys.executable, "train_dual.py",
    "--dataroot", "./datasets/mouse_kidney_dual_BF_AF_HE_256",
    "--model", "spif",
    "--dataset_mode", "dual_channel",
    "--input_nc", "2",
    "--output_nc", "3",
    "--lambda_identity", "0",
    "--batch_size", "8",
    "--n_epochs", "5",
    "--n_epochs_decay", "0",
    "--load_size", "286",
    "--crop_size", "256",
]


def run_one(n: int, name: str, lambda_low_freq: float):
    args = BASE_ARGS + [
        "--name", name,
        "--lambda_low_freq", str(lambda_low_freq),
    ]
    print(f"\n{'=' * 60}")
    print(f"Run {n}/20: {'WITH' if lambda_low_freq > 0 else 'NO'} low-freq  (lambda={lambda_low_freq})")
    print(f"  name={name}")
    print(f"{'=' * 60}")
    result = subprocess.run(args, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"[ERROR] Run {n} failed with code {result.returncode}")


if __name__ == "__main__":
    for i in range(1, 11):
        run_one(i, f"mouse_kidney_dual_spif_256_test_{i}", lambda_low_freq=0.0)

    for i in range(11, 21):
        run_one(i, f"mouse_kidney_dual_spif_256_test_{i}", lambda_low_freq=0.5)

    print("\nAll 20 runs complete.")
