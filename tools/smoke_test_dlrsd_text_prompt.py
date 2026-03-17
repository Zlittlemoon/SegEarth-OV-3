#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("SAM3 DLRSD smoke test")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to smoke yaml config",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="sam3/train/train.py",
        help="Path to official SAM3 training entry",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]

    env["SAM3_RS_FREEZE"] = "1"
    env["SAM3_RS_TRAIN_MODE"] = "minimal"
    env["CUDA_LAUNCH_BLOCKING"] = env.get("CUDA_LAUNCH_BLOCKING", "1")
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        args.train_script,
        "-c",
        args.config,
        "--use-cluster",
        "0",
        "--num-gpus",
        "1",
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()