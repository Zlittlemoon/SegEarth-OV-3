import argparse
import csv
from pathlib import Path
from contextlib import nullcontext

import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from segearthov3_segmentor_original import get_cls_idx

# Import functions from the single-image verification script.
from verify_batch_parallel import (
    HEAD_KEYS,
    serial_outputs,
    batch_outputs,
    batch_outputs_with_serial_text,
)


def list_images(image_dir, max_images=None):
    image_dir = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])

    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    return paths


def semantic_metrics(serial, batch):
    if "semantic_seg" not in serial or "semantic_seg" not in batch:
        return {}

    s = serial["semantic_seg"].squeeze(1)  # [C, H, W]
    b = batch["semantic_seg"].squeeze(1)   # [C, H, W]

    if s.ndim != 3 or b.ndim != 3 or s.shape != b.shape or s.shape[0] < 2:
        return {}

    s_pred = s.argmax(dim=0)
    b_pred = b.argmax(dim=0)

    changed = s_pred != b_pred
    agree = (~changed).float().mean().item()
    changed_ratio = changed.float().mean().item()

    top2 = torch.topk(s, k=2, dim=0).values
    margin = top2[0] - top2[1]

    if changed.any():
        changed_margin = margin[changed].mean().item()
    else:
        changed_margin = 0.0

    max_abs = (serial["semantic_seg"] - batch["semantic_seg"]).abs().max().item()
    denom = serial["semantic_seg"].abs().max().item() + 1e-12
    rel = max_abs / denom

    return {
        "semantic_pixel_agree": agree,
        "semantic_changed_ratio": changed_ratio,
        "semantic_changed_margin": changed_margin,
        "semantic_max_abs": max_abs,
        "semantic_rel": rel,
    }


def instance_metrics(serial, batch):
    if "pred_logits" not in serial or "pred_logits" not in batch:
        return {}

    s_logit = serial["pred_logits"].squeeze(-1)  # [C, Q]
    b_logit = batch["pred_logits"].squeeze(-1)   # [C, Q]

    if s_logit.shape != b_logit.shape:
        return {}

    s_top = s_logit.argmax(dim=1)
    b_top = b_logit.argmax(dim=1)

    top_agree = (s_top == b_top).float().mean().item()

    out = {
        "instance_top_query_agree": top_agree,
        "instance_top_changed": int((s_top != b_top).sum().item()),
    }

    max_abs_logits = (serial["pred_logits"] - batch["pred_logits"]).abs().max().item()
    denom_logits = serial["pred_logits"].abs().max().item() + 1e-12
    out["pred_logits_max_abs"] = max_abs_logits
    out["pred_logits_rel"] = max_abs_logits / denom_logits

    if "pred_masks" in serial and "pred_masks" in batch:
        if serial["pred_masks"].shape == batch["pred_masks"].shape:
            c = s_logit.shape[0]
            batch_idx = torch.arange(c)

            s_mask = serial["pred_masks"][batch_idx, s_top] > 0
            b_mask = batch["pred_masks"][batch_idx, b_top] > 0

            mask_agree = (s_mask == b_mask).float().mean().item()

            max_abs_masks = (serial["pred_masks"] - batch["pred_masks"]).abs().max().item()
            denom_masks = serial["pred_masks"].abs().max().item() + 1e-12

            out["instance_selected_mask_agree"] = mask_agree
            out["pred_masks_max_abs"] = max_abs_masks
            out["pred_masks_rel"] = max_abs_masks / denom_masks

    return out


def collect_metrics(serial, batch):
    out = {}
    out.update(semantic_metrics(serial, batch))
    out.update(instance_metrics(serial, batch))

    if "presence_logit_dec" in serial and "presence_logit_dec" in batch:
        max_abs = (serial["presence_logit_dec"] - batch["presence_logit_dec"]).abs().max().item()
        denom = serial["presence_logit_dec"].abs().max().item() + 1e-12
        out["presence_max_abs"] = max_abs
        out["presence_rel"] = max_abs / denom

    return out


def print_summary(rows, pair_name):
    sub = [r for r in rows if r["pair"] == pair_name]
    if not sub:
        return

    def vals(key):
        return [float(r[key]) for r in sub if key in r and r[key] != ""]

    print(f"\n== Summary: {pair_name} over {len(sub)} images ==")

    for key in [
        "semantic_pixel_agree",
        "semantic_changed_ratio",
        "instance_top_query_agree",
        "instance_selected_mask_agree",
    ]:
        v = vals(key)
        if not v:
            continue

        mean_v = sum(v) / len(v)
        min_v = min(v)
        max_v = max(v)

        if "agree" in key:
            print(f"  {key}: mean={mean_v * 100:.4f}% min={min_v * 100:.4f}% max={max_v * 100:.4f}%")
        else:
            print(f"  {key}: mean={mean_v * 100:.4f}% min={min_v * 100:.4f}% max={max_v * 100:.4f}%")

    v = vals("instance_top_changed")
    if v:
        print(f"  instance_top_changed: total={int(sum(v))} max_per_image={int(max(v))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="data/Potsdam/img_dir/val")
    ap.add_argument("--classes", default="configs/cls_potsdam.txt")
    ap.add_argument("--num", type=int, default=999)
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument(
        "--stage",
        choices=["semantic", "presence", "instance", "all"],
        default="all"
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    ap.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    ap.add_argument("--bpe", default="./sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    ap.add_argument("--isolate-text", action="store_true")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if (device == "cuda" and dtype == torch.bfloat16)
        else nullcontext()
    )

    image_paths = list_images(args.image_dir, args.max_images)

    query_words, query_idx = get_cls_idx(args.classes)
    prompts = query_words[: args.num]

    stages = ["semantic", "presence", "instance"] if args.stage == "all" else [args.stage]
    keys = [k for s in stages for k in HEAD_KEYS[s]]

    print(f"Image dir: {args.image_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"device={device} dtype={args.dtype}\n")

    model = build_sam3_image_model(
        bpe_path=args.bpe,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.0, device=device)

    rows = []

    for idx, image_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] {image_path}")

        image = Image.open(image_path).convert("RGB")

        serial = serial_outputs(processor, image, prompts, keys, autocast)
        batch = batch_outputs(processor, image, prompts, keys, autocast)

        m_ab = collect_metrics(serial, batch)
        row_ab = {
            "image_path": str(image_path),
            "pair": "A_vs_B",
            **m_ab,
        }
        rows.append(row_ab)

        if args.isolate_text:
            batch_serial_text = batch_outputs_with_serial_text(
                processor=processor,
                image=image,
                prompts=prompts,
                keys=keys,
                autocast=autocast,
            )
            m_ac = collect_metrics(serial, batch_serial_text)
            row_ac = {
                "image_path": str(image_path),
                "pair": "A_vs_C",
                **m_ac,
            }
            rows.append(row_ac)

        if "semantic_pixel_agree" in m_ab:
            print(
                f"  A_vs_B semantic agree={m_ab['semantic_pixel_agree'] * 100:.4f}% "
                f"changed={m_ab['semantic_changed_ratio'] * 100:.4f}%"
            )
        if "instance_top_query_agree" in m_ab:
            print(
                f"  A_vs_B instance top agree={m_ab['instance_top_query_agree'] * 100:.2f}% "
                f"selected mask agree={m_ab.get('instance_selected_mask_agree', 0.0) * 100:.4f}%"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(rows, "A_vs_B")
    if args.isolate_text:
        print_summary(rows, "A_vs_C")

    if args.out_csv is not None:
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()