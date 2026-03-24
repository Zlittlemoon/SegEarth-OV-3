#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check SAM3 semantic JSON for DLRSD / iSAID style datasets.

Expected JSON structure:
{
  "images": [...],
  "annotations": [...],
  "queries": [...],          # optional but expected for your current pipeline
  "categories": [...]
}

Main checks:
1. top-level structure
2. unique ids
3. image file existence + width/height consistency
4. annotation references and COCO RLE decode
5. query references: image_id and object_ids_output
6. decoded mask shape / area / bbox sanity
7. category / query statistics
8. optional visualization overlays

Usage examples:
python tools/check_sam3_semantic_json.py \
  --json data/sam3_semantic/dlrsd/train.sam3.semantic.json \
  --img-root data/DLRSD_split/train/imgs

python tools/check_sam3_semantic_json.py \
  --json data/sam3_semantic/isaid/val.sam3.semantic.json \
  --img-root data/iSAID_split/val/imgs \
  --visualize-n 20 \
  --vis-out outputs/json_check_vis/isaid_val

python tools/check_sam3_semantic_json.py \
  --json data/sam3_semantic/dlrsd/train.sam3.semantic.json \
  --img-root data/DLRSD_split/train/imgs \
  --fail-on-error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise ImportError(
        "pycocotools is required. Install with: pip install pycocotools"
    ) from exc


@dataclass
class CheckStats:
    n_images: int = 0
    n_annotations: int = 0
    n_queries: int = 0
    n_categories: int = 0

    n_missing_images: int = 0
    n_bad_image_size: int = 0
    n_bad_rle: int = 0
    n_bad_refs: int = 0
    n_empty_masks: int = 0
    n_bad_bbox: int = 0
    n_bad_area: int = 0
    n_bad_query_refs: int = 0
    n_query_without_ann: int = 0

    def has_error(self) -> bool:
        return any(
            x > 0
            for x in [
                self.n_missing_images,
                self.n_bad_image_size,
                self.n_bad_rle,
                self.n_bad_refs,
                self.n_bad_bbox,
                self.n_bad_area,
                self.n_bad_query_refs,
                self.n_query_without_ann,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Check SAM3 semantic JSON")
    parser.add_argument("--json", type=Path, required=True, help="Path to SAM3 semantic JSON")
    parser.add_argument("--img-root", type=Path, required=True, help="Root directory of images")
    parser.add_argument(
        "--visualize-n",
        type=int,
        default=0,
        help="Number of samples to visualize randomly (0 disables visualization)",
    )
    parser.add_argument(
        "--vis-out",
        type=Path,
        default=Path("outputs/json_check_vis"),
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for visualization sampling",
    )
    parser.add_argument(
        "--max-print-errors",
        type=int,
        default=50,
        help="Maximum number of detailed error lines to print",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero code if any hard error is found",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_top_keys(data: Dict[str, Any]) -> None:
    required = ["images", "annotations", "categories"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing top-level key: {k}")
    if "queries" not in data:
        print("[WARN] Top-level key 'queries' is missing. Continuing anyway.")


def build_id_index(items: Sequence[Dict[str, Any]], name: str) -> Dict[Any, Dict[str, Any]]:
    idx: Dict[Any, Dict[str, Any]] = {}
    dup_ids = []
    for x in items:
        if "id" not in x:
            raise KeyError(f"{name} item missing 'id': {x}")
        _id = x["id"]
        if _id in idx:
            dup_ids.append(_id)
        idx[_id] = x
    if dup_ids:
        raise ValueError(f"Duplicate ids found in {name}: {dup_ids[:20]}")
    return idx


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def decode_rle(segmentation: Any) -> np.ndarray:
    """
    Decode COCO RLE. Expects compressed RLE dict or uncompressed compatible format.
    Returns HxW uint8 mask.
    """
    if not isinstance(segmentation, dict):
        raise TypeError(f"Expected segmentation dict (RLE), got {type(segmentation)}")

    if "counts" not in segmentation or "size" not in segmentation:
        raise KeyError(f"RLE segmentation must contain 'counts' and 'size', got keys={list(segmentation.keys())}")

    rle = dict(segmentation)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        # pycocotools can return HxWxN; for your current semantic json it should be one mask
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = np.any(mask.astype(bool), axis=2).astype(np.uint8)
    return mask.astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def mask_bbox_xywh(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    return [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]


def approx_equal(a: float, b: float, atol: float = 1.0, rtol: float = 0.05) -> bool:
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def draw_overlay(
    img: Image.Image,
    mask: np.ndarray,
    bbox: Optional[Sequence[float]] = None,
    text_lines: Optional[List[str]] = None,
) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img).copy()

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.35

    mask_bool = mask.astype(bool)
    arr[mask_bool] = ((1 - alpha) * arr[mask_bool] + alpha * overlay_color).astype(np.uint8)

    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)

    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + max(0, w - 1), y + max(0, h - 1)], outline=(0, 255, 0), width=2)

    if text_lines:
        x0, y0 = 10, 10
        for i, line in enumerate(text_lines):
            draw.text((x0, y0 + i * 16), line, fill=(255, 255, 0))

    return out


def summarize_counter(title: str, counter: Counter, topk: int = 20) -> None:
    print(f"\n[{title}]")
    for k, v in counter.most_common(topk):
        print(f"  {k}: {v}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data = load_json(args.json)
    ensure_top_keys(data)

    images = data["images"]
    annotations = data["annotations"]
    queries = data.get("queries", [])
    categories = data["categories"]

    stats = CheckStats(
        n_images=len(images),
        n_annotations=len(annotations),
        n_queries=len(queries),
        n_categories=len(categories),
    )

    print("========== SAM3 Semantic JSON Check ==========")
    print(f"JSON      : {args.json}")
    print(f"IMG ROOT  : {args.img_root}")
    print(f"images    : {stats.n_images}")
    print(f"annos     : {stats.n_annotations}")
    print(f"queries   : {stats.n_queries}")
    print(f"categories: {stats.n_categories}")

    image_by_id = build_id_index(images, "images")
    ann_by_id = build_id_index(annotations, "annotations")
    query_by_id = build_id_index(queries, "queries") if queries else {}

    cat_name_by_id = {c["id"]: c.get("name", str(c["id"])) for c in categories}

    per_cat_ann = Counter()
    per_cat_query = Counter()
    per_query_text = Counter()
    per_image_ann = Counter()
    per_image_query = Counter()

    errors: List[str] = []
    decoded_ann_masks: Dict[Any, np.ndarray] = {}

    # -------------------------------------------------
    # Check images
    # -------------------------------------------------
    print("\n[1/5] Checking images...")
    for img in images:
        img_id = img["id"]
        file_name = img.get("file_name")
        width = img.get("width")
        height = img.get("height")

        if file_name is None:
            stats.n_bad_refs += 1
            errors.append(f"[IMAGE] missing file_name for image_id={img_id}")
            continue

        img_path = args.img_root / file_name
        if not img_path.exists():
            stats.n_missing_images += 1
            errors.append(f"[IMAGE] missing file: image_id={img_id}, path={img_path}")
            continue

        pil = safe_open_image(img_path)
        if pil is None:
            stats.n_missing_images += 1
            errors.append(f"[IMAGE] failed to open image: image_id={img_id}, path={img_path}")
            continue

        real_w, real_h = pil.size
        if width != real_w or height != real_h:
            stats.n_bad_image_size += 1
            errors.append(
                f"[IMAGE] size mismatch image_id={img_id}, json=({width},{height}), real=({real_w},{real_h}), path={img_path}"
            )

    # -------------------------------------------------
    # Check annotations
    # -------------------------------------------------
    print("[2/5] Checking annotations...")
    for ann in annotations:
        ann_id = ann["id"]
        image_id = ann.get("image_id")
        if image_id not in image_by_id:
            stats.n_bad_refs += 1
            errors.append(f"[ANN] bad image_id ref: ann_id={ann_id}, image_id={image_id}")
            continue

        img_info = image_by_id[image_id]
        W, H = img_info["width"], img_info["height"]

        category_id = ann.get("category_id", None)
        if category_id is not None:
            per_cat_ann[cat_name_by_id.get(category_id, str(category_id))] += 1
        per_image_ann[image_id] += 1

        seg = ann.get("segmentation", None)
        if seg is None:
            stats.n_bad_rle += 1
            errors.append(f"[ANN] missing segmentation: ann_id={ann_id}")
            continue

        try:
            mask = decode_rle(seg)
            decoded_ann_masks[ann_id] = mask
        except Exception as e:
            stats.n_bad_rle += 1
            errors.append(f"[ANN] bad RLE ann_id={ann_id}: {repr(e)}")
            continue

        if mask.shape != (H, W):
            stats.n_bad_rle += 1
            errors.append(
                f"[ANN] mask shape mismatch ann_id={ann_id}, decoded={mask.shape}, expected=({H},{W})"
            )
            continue

        area_decoded = mask_area(mask)
        area_json = ann.get("area", None)
        bbox_json = ann.get("bbox", None)
        bbox_decoded = mask_bbox_xywh(mask)

        if area_decoded == 0:
            stats.n_empty_masks += 1

        if area_json is not None and not approx_equal(float(area_json), float(area_decoded), atol=2.0, rtol=0.05):
            stats.n_bad_area += 1
            errors.append(
                f"[ANN] area mismatch ann_id={ann_id}, json={area_json}, decoded={area_decoded}"
            )

        if bbox_json is not None:
            if len(bbox_json) != 4:
                stats.n_bad_bbox += 1
                errors.append(f"[ANN] invalid bbox length ann_id={ann_id}, bbox={bbox_json}")
            else:
                for a, b in zip(bbox_json, bbox_decoded):
                    if not approx_equal(float(a), float(b), atol=2.0, rtol=0.05):
                        stats.n_bad_bbox += 1
                        errors.append(
                            f"[ANN] bbox mismatch ann_id={ann_id}, json={bbox_json}, decoded={bbox_decoded}"
                        )
                        break

    # -------------------------------------------------
    # Check queries
    # -------------------------------------------------
    print("[3/5] Checking queries...")
    for q in queries:
        qid = q["id"]
        image_id = q.get("image_id")
        if image_id not in image_by_id:
            stats.n_bad_query_refs += 1
            errors.append(f"[QUERY] bad image ref: query_id={qid}, image_id={image_id}")
            continue

        per_image_query[image_id] += 1

        query_text = q.get("query_text", q.get("text_input", ""))
        per_query_text[query_text] += 1

        original_cat_id = q.get("original_cat_id", None)
        if original_cat_id is not None:
            per_cat_query[cat_name_by_id.get(original_cat_id, str(original_cat_id))] += 1

        obj_ids = q.get("object_ids_output", [])
        if not isinstance(obj_ids, list):
            stats.n_bad_query_refs += 1
            errors.append(f"[QUERY] object_ids_output must be list, query_id={qid}")
            continue

        if len(obj_ids) == 0:
            stats.n_query_without_ann += 1
            errors.append(f"[QUERY] no object_ids_output: query_id={qid}, image_id={image_id}, text={query_text}")

        for oid in obj_ids:
            if oid not in ann_by_id:
                stats.n_bad_query_refs += 1
                errors.append(f"[QUERY] bad ann ref: query_id={qid}, ann_id={oid}")
                continue
            ann = ann_by_id[oid]
            if ann["image_id"] != image_id:
                stats.n_bad_query_refs += 1
                errors.append(
                    f"[QUERY] query-ann image mismatch: query_id={qid}, query.image_id={image_id}, ann_id={oid}, ann.image_id={ann['image_id']}"
                )

    # -------------------------------------------------
    # Aggregate sanity checks
    # -------------------------------------------------
    print("[4/5] Aggregating statistics...")
    n_images_wo_ann = sum(1 for img_id in image_by_id if per_image_ann[img_id] == 0)
    n_images_wo_query = sum(1 for img_id in image_by_id if per_image_query[img_id] == 0)

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    if args.visualize_n > 0:
        print("[5/5] Writing visualization samples...")
        args.vis_out.mkdir(parents=True, exist_ok=True)

        valid_queries = []
        for q in queries:
            obj_ids = q.get("object_ids_output", [])
            image_id = q.get("image_id", None)
            if image_id not in image_by_id:
                continue
            if len(obj_ids) == 0:
                continue
            ok = True
            for oid in obj_ids:
                if oid not in decoded_ann_masks:
                    ok = False
                    break
            if ok:
                valid_queries.append(q)

        n_vis = min(args.visualize_n, len(valid_queries))
        sampled = random.sample(valid_queries, n_vis) if n_vis > 0 else []

        for i, q in enumerate(sampled):
            image_id = q["image_id"]
            img_info = image_by_id[image_id]
            img_path = args.img_root / img_info["file_name"]
            pil = safe_open_image(img_path)
            if pil is None:
                continue

            obj_ids = q.get("object_ids_output", [])
            union_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            for oid in obj_ids:
                union_mask = np.logical_or(union_mask, decoded_ann_masks[oid] > 0)
            union_mask = union_mask.astype(np.uint8)

            bbox = mask_bbox_xywh(union_mask)
            text_lines = [
                f"idx={i}",
                f"image_id={image_id}",
                f"query_id={q['id']}",
                f"file={img_info['file_name']}",
                f"text={q.get('query_text', q.get('text_input', ''))}",
                f"original_cat_id={q.get('original_cat_id', 'NA')}",
                f"mask_area={mask_area(union_mask)}",
                f"num_obj_ids={len(obj_ids)}",
            ]
            vis = draw_overlay(pil, union_mask, bbox=bbox, text_lines=text_lines)
            out_path = args.vis_out / f"{i:04d}_img{image_id}_q{q['id']}.png"
            vis.save(out_path)

        print(f"Visualization written to: {args.vis_out}")

    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------
    print("\n========== Summary ==========")
    print(f"images                : {stats.n_images}")
    print(f"annotations           : {stats.n_annotations}")
    print(f"queries               : {stats.n_queries}")
    print(f"categories            : {stats.n_categories}")
    print(f"missing image files   : {stats.n_missing_images}")
    print(f"image size mismatch   : {stats.n_bad_image_size}")
    print(f"bad RLE               : {stats.n_bad_rle}")
    print(f"bad image/ann refs    : {stats.n_bad_refs}")
    print(f"bad query refs        : {stats.n_bad_query_refs}")
    print(f"queries w/o ann       : {stats.n_query_without_ann}")
    print(f"empty decoded masks   : {stats.n_empty_masks}")
    print(f"bad bbox              : {stats.n_bad_bbox}")
    print(f"bad area              : {stats.n_bad_area}")
    print(f"images w/o ann        : {n_images_wo_ann}")
    print(f"images w/o query      : {n_images_wo_query}")

    summarize_counter("Annotations per category", per_cat_ann)
    summarize_counter("Queries per category", per_cat_query)
    summarize_counter("Top query texts", per_query_text)

    if errors:
        print(f"\n========== First {min(len(errors), args.max_print_errors)} Errors ==========")
        for e in errors[: args.max_print_errors]:
            print(e)

    if stats.has_error():
        print("\n[RESULT] HARD ERRORS FOUND")
        if args.fail_on_error:
            sys.exit(2)
    else:
        print("\n[RESULT] No hard errors found")

    # soft warnings
    if stats.n_empty_masks > 0:
        print(f"[WARN] Found {stats.n_empty_masks} empty decoded masks.")
    if n_images_wo_ann > 0:
        print(f"[WARN] Found {n_images_wo_ann} images without annotations.")
    if n_images_wo_query > 0:
        print(f"[WARN] Found {n_images_wo_query} images without queries.")


if __name__ == "__main__":
    main()