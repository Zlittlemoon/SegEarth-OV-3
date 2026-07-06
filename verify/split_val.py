"""
Split the validation image list into calibration and evaluation halves.

Splitting criterion: filename hash parity (odd/even on last hex digit of md5).
This is deterministic and reproducible without storing a seed — anyone with
the file list can reproduce the split.

Outputs two text files (one image path per line):
  <out_dir>/cal_images.txt
  <out_dir>/eval_images.txt

Usage:
    python verify/split_val.py configs/cfg_openearthmap.py --out-dir verify/splits/oem
    python verify/split_val.py configs/cfg_loveda.py      --out-dir verify/splits/loveda
"""

import argparse
import hashlib
import os

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

import mmseg.datasets
import segearthov3_segmentor_merge
import custom_datasets


def file_hash_parity(path: str) -> int:
    """Return 0 (cal) or 1 (eval) based on last hex digit of md5(basename)."""
    name = os.path.basename(path)
    digest = hashlib.md5(name.encode()).hexdigest()
    return int(digest[-1], 16) % 2


def parse_args():
    ap = argparse.ArgumentParser(description="Split val set into cal/eval halves by filename hash parity")
    ap.add_argument("config")
    ap.add_argument("--out-dir", required=True, help="Directory to write cal_images.txt and eval_images.txt")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    loader = Runner.build_dataloader(cfg.test_dataloader)
    dataset = loader.dataset

    cal_paths = []
    eval_paths = []

    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        img_path = info["img_path"]
        if file_hash_parity(img_path) == 0:
            cal_paths.append(img_path)
        else:
            eval_paths.append(img_path)

    os.makedirs(args.out_dir, exist_ok=True)
    cal_file = os.path.join(args.out_dir, "cal_images.txt")
    eval_file = os.path.join(args.out_dir, "eval_images.txt")

    with open(cal_file, "w") as f:
        f.write("\n".join(cal_paths) + "\n")
    with open(eval_file, "w") as f:
        f.write("\n".join(eval_paths) + "\n")

    total = len(cal_paths) + len(eval_paths)
    print(f"Total images : {total}")
    print(f"Cal  (parity=0): {len(cal_paths)}  ({len(cal_paths)/total:.1%})  -> {cal_file}")
    print(f"Eval (parity=1): {len(eval_paths)}  ({len(eval_paths)/total:.1%})  -> {eval_file}")


if __name__ == "__main__":
    main()
