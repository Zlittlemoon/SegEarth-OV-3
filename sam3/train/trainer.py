# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy
import contextlib
import fnmatch
import gc
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.train.utils.freeze_rs_text_prompt import maybe_apply_remote_sensing_freeze
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.optim.optimizer import construct_optimizer

from sam3.train.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)

from sam3.train.utils.distributed import all_reduce_max, barrier, get_rank

from sam3.train.utils.logger import Logger, setup_logging
from sam3.train.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"

def _unwrap_stage_output(find_stages):
    if isinstance(find_stages, dict):
        return find_stages
    if isinstance(find_stages, list):
        x = find_stages[-1]
        if isinstance(x, list):
            x = x[-1]
        if isinstance(x, dict):
            return x
    # SAM3Output 通常也支持 len()/索引
    if hasattr(find_stages, "__len__") and len(find_stages) > 0:
        x = find_stages[-1]
        if isinstance(x, list):
            x = x[-1]
        if isinstance(x, dict):
            return x
    raise TypeError(f"Cannot unwrap find_stages of type={type(find_stages)}")


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def _tensor_brief(x):
    x = _to_cpu(x)
    if not torch.is_tensor(x):
        return {"type": str(type(x))}
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
    }
    if x.numel() > 0:
        if x.dtype == torch.bool:
            out["true_ratio"] = float(x.float().mean().item())
        else:
            out["min"] = float(x.min().item())
            out["max"] = float(x.max().item())
            out["mean"] = float(x.float().mean().item())
    return out


def _summarize_obj(obj, depth=2):
    if depth < 0:
        return str(type(obj))
    if torch.is_tensor(obj):
        return _tensor_brief(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _summarize_obj(v, depth - 1) for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "len": len(obj),
            "items": [_summarize_obj(v, depth - 1) for v in list(obj)[:5]],
        }
    if hasattr(obj, "__dict__"):
        return {
            "type": type(obj).__name__,
            "attrs": {
                k: _summarize_obj(v, depth - 1)
                for k, v in list(vars(obj).items())[:30]
            },
        }
    return str(type(obj))


def _denorm_img(img):
    # 你的 dataset 当前是 mean=0.5/std=0.5 的 [-1,1] 规范化
    img = img.float()
    img = img * 0.5 + 0.5
    return img.clamp(0.0, 1.0)


def _debug_enabled() -> bool:
    return (
        os.getenv("SAM3_RS_DEBUG", "0") == "1"
        or os.getenv("SAM3_RS_DEBUG_PROMPT", "0") == "1"
    )


def _debug_rank0_enabled(rank: Optional[int] = None) -> bool:
    if not _debug_enabled():
        return False
    if rank is None:
        return True
    return int(rank) == 0


def _debug_print(*args, rank: Optional[int] = None, **kwargs):
    if _debug_rank0_enabled(rank):
        print(*args, **kwargs)


def _fg_ratio_per_sample(x):
    x = _to_cpu(x)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim < 2:
        return x.float()
    return x.float().flatten(1).mean(1)


_DEBUG_JSON_LOOKUP_CACHE = {}


def _slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def _resolve_query_text_from_json_query(q: Dict[str, Any]) -> str:
    for k in ["query_text", "text", "text_input", "queried_category"]:
        if k in q and q[k] is not None:
            return str(q[k])
    return "unknown"


def _resolve_ann_ids_from_json_query(q: Dict[str, Any]) -> List[int]:
    if "object_ids_output" in q:
        obj_ids = q["object_ids_output"]
        if obj_ids is None:
            return []
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        return [int(x) for x in obj_ids if x is not None]

    if "annotation_id" in q:
        ann_id = q["annotation_id"]
        if ann_id is None:
            return []
        return [int(ann_id)]

    return []


def _get_dataset_for_phase(trainer, phase: str):
    if phase == Phase.TRAIN:
        return getattr(trainer, "train_dataset", None)
    return getattr(trainer, "val_dataset", None)


def _discover_ann_file_and_img_root(dataset):
    candidates = [dataset]
    for attr in ["dataset", "base_dataset", "_dataset"]:
        if hasattr(dataset, attr):
            candidates.append(getattr(dataset, attr))

    for ds in candidates:
        ann_file = getattr(ds, "ann_file", None)
        img_folder = getattr(ds, "img_folder", None)
        if ann_file is not None and img_folder is not None:
            return str(ann_file), str(img_folder)
    return None, None


def _get_or_build_json_lookup(ann_file: Optional[str]):
    if ann_file is None:
        return None

    ann_file = str(ann_file)
    if ann_file in _DEBUG_JSON_LOOKUP_CACHE:
        return _DEBUG_JSON_LOOKUP_CACHE[ann_file]

    try:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[DEBUG] failed to read ann_file for debug naming: {ann_file} | {e}")
        _DEBUG_JSON_LOOKUP_CACHE[ann_file] = None
        return None

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    queries = data.get("queries", [])

    image_by_id = {}
    for img in images:
        if "id" in img:
            image_by_id[int(img["id"])] = img

    ann_by_id = {}
    for ann in annotations:
        if "id" in ann:
            ann_by_id[int(ann["id"])] = ann

    qid_by_first_key = {}
    qid_by_text_img = {}

    query_by_id = {}
    qmeta_by_first_key = {}
    qmeta_by_text_img = {}
    qmeta_by_ann_key = {}
    qmeta_by_imgcat = {}

    for i, q in enumerate(queries):
        ann_ids = _resolve_ann_ids_from_json_query(q)
        if len(ann_ids) == 0:
            continue

        query_text = _resolve_query_text_from_json_query(q)
        qid = int(q.get("id", i))
        original_cat_id = int(q.get("original_cat_id", -1)) if q.get("original_cat_id") is not None else -1

        image_id = q.get("image_id", None)
        if image_id is None:
            image_ids = list({int(ann_by_id[aid]["image_id"]) for aid in ann_ids if aid in ann_by_id})
            if len(image_ids) == 1:
                image_id = image_ids[0]
        if image_id is None:
            continue
        image_id = int(image_id)

        first_ann_id = int(ann_ids[0])
        first_ann = ann_by_id.get(first_ann_id, {})
        file_name = image_by_id.get(image_id, {}).get("file_name")

        meta = {
            "query_id": qid,
            "image_id": image_id,
            "query_text": query_text,
            "original_category_id": original_cat_id,
            "first_ann_id": first_ann_id,
            "raw_input_box": q.get("input_box"),
            "raw_input_box_label": q.get("input_box_label"),
            "raw_input_points": q.get("input_points"),
            "first_ann_bbox": first_ann.get("bbox"),
            "file_name": file_name,
        }

        query_by_id[qid] = meta

        key1 = (image_id, query_text, first_ann_id, original_cat_id)
        qid_by_first_key.setdefault(key1, []).append(qid)
        qmeta_by_first_key.setdefault(key1, []).append(meta)

        key2 = (image_id, query_text, original_cat_id)
        qid_by_text_img.setdefault(key2, []).append(qid)
        qmeta_by_text_img.setdefault(key2, []).append(meta)

        key3 = (image_id, first_ann_id, original_cat_id)
        qmeta_by_ann_key.setdefault(key3, []).append(meta)

        key4 = (image_id, original_cat_id)
        qmeta_by_imgcat.setdefault(key4, []).append(meta)

    lookup = {
        "ann_file": ann_file,
        "image_by_id": image_by_id,
        "ann_by_id": ann_by_id,
        "qid_by_first_key": qid_by_first_key,
        "qid_by_text_img": qid_by_text_img,
        "query_by_id": query_by_id,
        "qmeta_by_first_key": qmeta_by_first_key,
        "qmeta_by_text_img": qmeta_by_text_img,
        "qmeta_by_ann_key": qmeta_by_ann_key,
        "qmeta_by_imgcat": qmeta_by_imgcat,
    }
    _DEBUG_JSON_LOOKUP_CACHE[ann_file] = lookup
    return lookup


def _try_get_original_image_ids(batch):
    if hasattr(batch, "find_metadatas") and len(batch.find_metadatas) > 0:
        meta = batch.find_metadatas[0]
        if isinstance(meta, dict) and "original_image_id" in meta:
            return _to_cpu(meta["original_image_id"]).view(-1)
        if hasattr(meta, "original_image_id"):
            return _to_cpu(meta.original_image_id).view(-1)
    return None


def _try_get_original_category_ids(batch):
    if hasattr(batch, "find_metadatas") and len(batch.find_metadatas) > 0:
        meta = batch.find_metadatas[0]
        if isinstance(meta, dict) and "original_category_id" in meta:
            return _to_cpu(meta["original_category_id"]).view(-1)
        if hasattr(meta, "original_category_id"):
            return _to_cpu(meta.original_category_id).view(-1)
    return None


def _try_get_object_ids(find_targets, batch):
    if isinstance(find_targets, list) and len(find_targets) > 0 and isinstance(find_targets[0], dict):
        t0 = find_targets[0]
        for k in ["object_ids_padded", "object_ids"]:
            if k in t0 and torch.is_tensor(t0[k]):
                v = _to_cpu(t0[k])
                if v.ndim == 2 and v.shape[1] >= 1:
                    return v[:, 0].view(-1)
                return v.view(-1)

    if hasattr(batch, "find_metadatas") and len(batch.find_metadatas) > 0:
        meta = batch.find_metadatas[0]
        if isinstance(meta, dict) and "object_id" in meta:
            return _to_cpu(meta["object_id"]).view(-1)
        if hasattr(meta, "object_id"):
            return _to_cpu(meta.object_id).view(-1)
    return None


def _resolve_debug_identity(i, text, image_ids, object_ids, original_cat_ids, json_lookup):
    image_id = None
    object_id = None
    original_cat_id = -1
    file_name = None
    query_id = None

    if image_ids is not None and i < len(image_ids):
        image_id = int(image_ids[i].item())

    if object_ids is not None and i < len(object_ids):
        object_id = int(object_ids[i].item())

    if original_cat_ids is not None and i < len(original_cat_ids):
        original_cat_id = int(original_cat_ids[i].item())

    if json_lookup is not None and image_id is not None:
        img_info = json_lookup["image_by_id"].get(int(image_id))
        if img_info is not None:
            file_name = img_info.get("file_name")

        if object_id is not None:
            key1 = (int(image_id), str(text), int(object_id), int(original_cat_id))
            qids = json_lookup["qid_by_first_key"].get(key1, [])
            if len(qids) == 1:
                query_id = int(qids[0])

        if query_id is None:
            key2 = (int(image_id), str(text), int(original_cat_id))
            qids = json_lookup["qid_by_text_img"].get(key2, [])
            if len(qids) == 1:
                query_id = int(qids[0])

    return {
        "image_id": image_id,
        "object_id": object_id,
        "original_category_id": original_cat_id,
        "file_name": file_name,
        "query_id": query_id,
    }


def _try_get_texts(batch):
    # 1) 正确读取 text bank
    if hasattr(batch, "find_text_batch"):
        text_bank = getattr(batch, "find_text_batch")
        if isinstance(text_bank, (list, tuple)):
            text_bank = [str(x) for x in text_bank]

            # 2) 用 text_ids 把每个样本映射回真实 text
            if hasattr(batch, "find_inputs") and len(batch.find_inputs) > 0:
                find_input = batch.find_inputs[0]
                if hasattr(find_input, "text_ids") and torch.is_tensor(find_input.text_ids):
                    text_ids = _to_cpu(find_input.text_ids).view(-1).tolist()
                    out = []
                    for tid in text_ids:
                        tid = int(tid)
                        if 0 <= tid < len(text_bank):
                            out.append(text_bank[tid])
                        else:
                            out.append(f"<bad text id {tid}>")
                    return out

            # fallback: 只有 text bank，没有 text_ids
            return text_bank

    # 旧逻辑保留作兜底
    for attr in ["query_texts", "texts", "find_texts"]:
        if hasattr(batch, attr):
            v = getattr(batch, attr)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]

    if hasattr(batch, "find_metadatas"):
        metas = getattr(batch, "find_metadatas")
        texts = []
        ok = False
        for m in metas:
            if isinstance(m, dict):
                t = m.get("query_text", m.get("text", None))
            else:
                t = getattr(m, "query_text", getattr(m, "text", None))
            texts.append(None if t is None else str(t))
            ok = ok or (t is not None)
        if ok:
            return texts

    return None


def _try_get_boxes(batch, find_targets):
    # 优先用真实 prompt box
    if hasattr(batch, "find_inputs") and len(batch.find_inputs) > 0:
        find_input = batch.find_inputs[0]
        if hasattr(find_input, "input_boxes") and torch.is_tensor(find_input.input_boxes):
            v = _to_cpu(find_input.input_boxes)
            if v.ndim == 3 and v.shape[0] == 1 and v.shape[-1] == 4:
                v = v[0]
            elif v.ndim == 3 and v.shape[1] == 1 and v.shape[-1] == 4:
                v = v[:, 0]
            return v, "batch.find_inputs[0].input_boxes"

    for attr in ["input_boxes", "find_input_boxes", "prompt_boxes", "boxes"]:
        if hasattr(batch, attr):
            v = getattr(batch, attr)
            if torch.is_tensor(v) and v.ndim >= 2:
                v = _to_cpu(v)
                if v.ndim == 3 and v.shape[0] == 1 and v.shape[-1] == 4:
                    v = v[0]
                elif v.ndim == 3 and v.shape[1] == 1 and v.shape[-1] == 4:
                    v = v[:, 0]
                return v, f"batch.{attr}"

    if isinstance(find_targets, list) and len(find_targets) > 0 and isinstance(find_targets[0], dict):
        t0 = find_targets[0]
        for k in ["boxes", "boxes_padded"]:
            if k in t0 and torch.is_tensor(t0[k]):
                v = _to_cpu(t0[k])
                if v.ndim == 3 and v.shape[1] == 1 and v.shape[-1] == 4:
                    v = v[:, 0]
                return v, f"find_targets[0]['{k}']"

    return None, None


def _try_get_original_sizes(batch):
    if hasattr(batch, "find_metadatas") and len(batch.find_metadatas) > 0:
        meta = batch.find_metadatas[0]
        if isinstance(meta, dict) and "original_size" in meta:
            return _to_cpu(meta["original_size"])
        if hasattr(meta, "original_size"):
            return _to_cpu(meta.original_size)
    return None


def _normalize_box_tensor(boxes):
    if boxes is None:
        return None
    boxes = _to_cpu(boxes)
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes)
    if boxes.ndim == 3 and boxes.shape[0] == 1 and boxes.shape[-1] == 4:
        boxes = boxes[0]
    elif boxes.ndim == 3 and boxes.shape[1] == 1 and boxes.shape[-1] == 4:
        boxes = boxes[:, 0]
    return boxes


def _normalize_box_mask(mask):
    if mask is None:
        return None
    mask = _to_cpu(mask)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    else:
        mask = mask.view(-1)
    return mask


def _try_get_prompt_boxes(batch):
    if hasattr(batch, "find_inputs") and len(batch.find_inputs) > 0:
        fi = batch.find_inputs[0]
        if hasattr(fi, "input_boxes") and torch.is_tensor(fi.input_boxes):
            boxes = _normalize_box_tensor(fi.input_boxes)
            masks = None
            if hasattr(fi, "input_boxes_mask") and torch.is_tensor(fi.input_boxes_mask):
                masks = _normalize_box_mask(fi.input_boxes_mask)
            return boxes, masks, "batch.find_inputs[0].input_boxes"
    return None, None, None


def _try_get_target_boxes_xywh(find_targets):
    if isinstance(find_targets, list) and len(find_targets) > 0 and isinstance(find_targets[0], dict):
        t0 = find_targets[0]
        if "boxes" in t0 and torch.is_tensor(t0["boxes"]):
            return _normalize_box_tensor(t0["boxes"]), "find_targets[0]['boxes']"
        if "boxes_padded" in t0 and torch.is_tensor(t0["boxes_padded"]):
            return _normalize_box_tensor(t0["boxes_padded"]), "find_targets[0]['boxes_padded']"
    return None, None


def _pick_item_box(boxes, i, masks=None):
    if boxes is None:
        return None
    if not torch.is_tensor(boxes):
        boxes = torch.as_tensor(boxes)
    if i >= boxes.shape[0]:
        return None
    b = boxes[i]
    if b.ndim == 1:
        return b.flatten().tolist()
    if b.ndim == 2 and b.shape[0] > 0:
        return b[0].flatten().tolist()
    return None


def _xywh_to_xyxy(box):
    if box is None:
        return None
    x, y, w, h = [float(v) for v in box[:4]]
    return [x, y, x + w, y + h]


def _scale_raw_json_xywh_to_current_image(raw_box, orig_size_hw, cur_h, cur_w):
    if raw_box is None or orig_size_hw is None:
        return None

    box = torch.as_tensor(raw_box, dtype=torch.float32).view(-1, 4)[0].tolist()
    oh, ow = int(orig_size_hw[0]), int(orig_size_hw[1])
    sx = float(cur_w) / float(ow)
    sy = float(cur_h) / float(oh)

    x, y, w, h = box
    return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]


def _maybe_scale_norm_xyxy(box_xyxy, h, w):
    if box_xyxy is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    return [x1, y1, x2, y2]


def _maybe_scale_norm_xywh(box_xywh, h, w):
    if box_xywh is None:
        return None
    x, y, bw, bh = [float(v) for v in box_xywh[:4]]
    if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.5:
        x *= w
        y *= h
        bw *= w
        bh *= h
    return [x, y, x + bw, y + bh]


def _draw_xyxy(ax, box_xyxy, color, label, linestyle="-", linewidth=2):
    if box_xyxy is None:
        return
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (x1, y1),
        w,
        h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)


def _resolve_query_meta(i, text, image_ids, object_ids, original_cat_ids, json_lookup):
    identity = _resolve_debug_identity(
        i=i,
        text=text,
        image_ids=image_ids,
        object_ids=object_ids,
        original_cat_ids=original_cat_ids,
        json_lookup=json_lookup,
    )

    meta = None
    if json_lookup is not None and identity.get("query_id") is not None:
        meta = json_lookup["query_by_id"].get(int(identity["query_id"]))

    image_id = identity.get("image_id")
    object_id = identity.get("object_id")
    original_cat_id = identity.get("original_category_id", -1)

    if meta is None and json_lookup is not None and image_id is not None and object_id is not None:
        if text is not None and text != "<text not found in batch>":
            cands = json_lookup["qmeta_by_first_key"].get(
                (int(image_id), str(text), int(object_id), int(original_cat_id)), []
            )
            if len(cands) == 1:
                meta = cands[0]

        if meta is None:
            cands = json_lookup["qmeta_by_ann_key"].get(
                (int(image_id), int(object_id), int(original_cat_id)), []
            )
            if len(cands) == 1:
                meta = cands[0]

    if meta is None and json_lookup is not None and image_id is not None:
        if text is not None and text != "<text not found in batch>":
            cands = json_lookup["qmeta_by_text_img"].get(
                (int(image_id), str(text), int(original_cat_id)), []
            )
            if len(cands) == 1:
                meta = cands[0]

        if meta is None:
            cands = json_lookup["qmeta_by_imgcat"].get(
                (int(image_id), int(original_cat_id)), []
            )
            if len(cands) == 1:
                meta = cands[0]

    if meta is None:
        meta = {}

    out = dict(identity)
    out["raw_input_box"] = meta.get("raw_input_box")
    out["first_ann_bbox"] = meta.get("first_ann_bbox")
    return out


def dump_trainable_params(model, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        rows.append(
            {
                "name": name,
                "shape": list(p.shape),
                "numel": int(n),
                "requires_grad": bool(p.requires_grad),
            }
        )

    with open(save_dir / "trainable_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_params": int(total),
                "trainable_params": int(trainable),
                "frozen_params": int(total - trainable),
                "trainable_param_names": [r["name"] for r in rows if r["requires_grad"]],
                "all_params": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DEBUG] saved trainable param summary -> {save_dir / 'trainable_params.json'}")
    print(f"[DEBUG] total={total}, trainable={trainable}, frozen={total - trainable}")
    print("[DEBUG] first 50 trainable names:")
    shown = 0
    for r in rows:
        if r["requires_grad"]:
            print("   ", r["name"])
            shown += 1
            if shown >= 50:
                break


def save_first_batch_visuals(batch, find_stages, find_targets, save_dir, max_items=8, ann_file=None, img_root=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_summary = {
        "batch_type": str(type(batch)),
        "batch_summary": _summarize_obj(batch, depth=2),
        "find_targets_summary": _summarize_obj(find_targets, depth=2),
    }
    with open(save_dir / "batch_structure.json", "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    json_lookup = _get_or_build_json_lookup(ann_file)
    image_ids = _try_get_original_image_ids(batch)
    original_cat_ids = _try_get_original_category_ids(batch)
    object_ids = _try_get_object_ids(find_targets, batch)

    out = _unwrap_stage_output(find_stages)
    if "semantic_seg" not in out:
        raise KeyError(f"semantic_seg not found, stage keys={list(out.keys())}")

    pred = _to_cpu(out["semantic_seg"]).float()

    if not hasattr(batch, "img_batch"):
        raise AttributeError("batch has no img_batch; please inspect batch_structure.json")
    imgs = _to_cpu(batch.img_batch)

    if not (isinstance(find_targets, list) and len(find_targets) > 0 and isinstance(find_targets[0], dict)):
        raise TypeError("find_targets format unexpected; please inspect batch_structure.json")

    target_dict = find_targets[0]
    if "semantic_masks" in target_dict:
        gt = _to_cpu(target_dict["semantic_masks"]).bool()
    elif "masks" in target_dict:
        gt = _to_cpu(target_dict["masks"]).bool()
    else:
        raise KeyError(f"Neither semantic_masks nor masks in find_targets[0]: {list(target_dict.keys())}")

    pred_up = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    prob_up = pred_up.sigmoid()
    pred_bin = (prob_up > 0.5).squeeze(1)

    texts = _try_get_texts(batch)

    prompt_boxes, prompt_box_masks, prompt_box_source = _try_get_prompt_boxes(batch)
    target_boxes_xywh, target_box_source = _try_get_target_boxes_xywh(find_targets)
    legacy_boxes, legacy_box_source = _try_get_boxes(batch, find_targets)
    original_sizes = _try_get_original_sizes(batch)

    runtime_dump = {
        "trainer_file": __file__,
        "ann_file": ann_file,
        "img_root": img_root,
        "texts_from_helper": texts,
        "prompt_box_source": prompt_box_source,
        "target_box_source": target_box_source,
        "legacy_box_source": legacy_box_source,
    }

    if hasattr(batch, "find_text_batch"):
        runtime_dump["find_text_batch"] = list(batch.find_text_batch)

    if hasattr(batch, "find_inputs") and len(batch.find_inputs) > 0:
        fi = batch.find_inputs[0]

        if hasattr(fi, "text_ids") and torch.is_tensor(fi.text_ids):
            runtime_dump["text_ids_shape"] = list(fi.text_ids.shape)
            runtime_dump["text_ids_head"] = _to_cpu(fi.text_ids).view(-1)[:32].tolist()

        if hasattr(fi, "input_boxes") and torch.is_tensor(fi.input_boxes):
            runtime_dump["input_boxes_shape"] = list(_to_cpu(fi.input_boxes).shape)
            runtime_dump["input_boxes_head"] = _to_cpu(fi.input_boxes)[:4].tolist()

        if hasattr(fi, "input_boxes_mask") and torch.is_tensor(fi.input_boxes_mask):
            runtime_dump["input_boxes_mask_shape"] = list(_to_cpu(fi.input_boxes_mask).shape)
            runtime_dump["input_boxes_mask_head"] = _to_cpu(fi.input_boxes_mask)[:4].tolist()

    runtime_dump["normalized_prompt_boxes_shape"] = None if prompt_boxes is None else list(prompt_boxes.shape)
    runtime_dump["normalized_prompt_boxes_head"] = None if prompt_boxes is None else prompt_boxes[:4].tolist()
    runtime_dump["normalized_target_boxes_shape"] = None if target_boxes_xywh is None else list(target_boxes_xywh.shape)
    runtime_dump["normalized_target_boxes_head"] = None if target_boxes_xywh is None else target_boxes_xywh[:4].tolist()

    with open(save_dir / "runtime_prompt_dump.json", "w", encoding="utf-8") as f:
        json.dump(runtime_dump, f, ensure_ascii=False, indent=2)

    B = min(int(imgs.shape[0]), int(gt.shape[0]), int(pred_bin.shape[0]), max_items)

    with open(save_dir / "batch_level_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "img_batch": _tensor_brief(imgs),
                "gt": _tensor_brief(gt),
                "pred": _tensor_brief(pred),
                "pred_up": _tensor_brief(pred_up),
                "pred_bin": _tensor_brief(pred_bin),
                "texts_found": texts is not None,
                "prompt_box_source": prompt_box_source,
                "target_box_source": target_box_source,
                "legacy_box_source": legacy_box_source,
                "ann_file": ann_file,
                "img_root": img_root,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    visual_rows = []

    for i in range(B):
        img = _denorm_img(imgs[i]).permute(1, 2, 0).numpy()
        gti = gt[i].numpy()
        pbi = pred_bin[i].numpy()
        probi = prob_up[i, 0].numpy()

        title_text = texts[i] if texts is not None and i < len(texts) else "<text not found in batch>"
        meta = _resolve_query_meta(
            i=i,
            text=title_text,
            image_ids=image_ids,
            object_ids=object_ids,
            original_cat_ids=original_cat_ids,
            json_lookup=json_lookup,
        )

        image_id = meta["image_id"]
        query_id = meta["query_id"]
        file_name = meta["file_name"]

        img_id_str = str(image_id) if image_id is not None else "UNK"
        qid_str = str(query_id) if query_id is not None else "UNK"
        debug_name = f"{i:04d}_{_slugify(title_text)}_img{img_id_str}_q{qid_str}.png"

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img)
        title_lines = [
            f"text={title_text}",
            f"img_id={img_id_str} qid={qid_str}",
        ]
        if file_name is not None:
            title_lines.append(f"file={file_name}")
        axes[0].set_title("\n".join(title_lines))
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].imshow(gti, alpha=0.45)
        axes[1].set_title(f"GT mask fg_ratio={gti.mean():.4f}")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(pbi, alpha=0.45)
        axes[2].set_title(f"Pred bin@0.5 fg_ratio={pbi.mean():.4f}")
        axes[2].axis("off")

        axes[3].imshow(probi, cmap="viridis")
        axes[3].set_title(f"Pred prob upsampled mean={probi.mean():.4f}")
        axes[3].axis("off")

        cur_h, cur_w = img.shape[0], img.shape[1]

        prompt_box_raw = _pick_item_box(prompt_boxes, i, prompt_box_masks)
        prompt_box_xyxy = _maybe_scale_norm_xywh(prompt_box_raw, cur_h, cur_w)

        target_box_raw = _pick_item_box(target_boxes_xywh, i, None)
        target_box_xyxy = _maybe_scale_norm_xywh(target_box_raw, cur_h, cur_w)

        orig_size_hw = None
        if original_sizes is not None and i < len(original_sizes):
            osz = original_sizes[i]
            if torch.is_tensor(osz):
                osz = osz.flatten().tolist()
            if len(osz) >= 2:
                orig_size_hw = (osz[0], osz[1])

        json_raw_input_box = meta.get("raw_input_box")
        json_box_xyxy = _scale_raw_json_xywh_to_current_image(
            json_raw_input_box,
            orig_size_hw,
            cur_h,
            cur_w,
        )

        fallback_ann_bbox = meta.get("first_ann_bbox")
        fallback_ann_xyxy = _scale_raw_json_xywh_to_current_image(
            fallback_ann_bbox,
            orig_size_hw,
            cur_h,
            cur_w,
        )

        _draw_xyxy(axes[0], prompt_box_xyxy, color="lime", label="prompt_box(batch)", linestyle="-", linewidth=2)
        _draw_xyxy(axes[0], target_box_xyxy, color="red", label="target_box(find_targets)", linestyle="-", linewidth=2)
        _draw_xyxy(axes[0], json_box_xyxy, color="cyan", label="json_input_box(raw->scaled)", linestyle="--", linewidth=2)

        if json_raw_input_box is None:
            _draw_xyxy(axes[0], fallback_ann_xyxy, color="yellow", label="ann_bbox_fallback", linestyle="--", linewidth=2)

        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            axes[0].legend(
                uniq.values(),
                uniq.keys(),
                loc="lower left",
                fontsize=8,
                framealpha=0.8,
            )

        fig.tight_layout()
        fig.savefig(save_dir / debug_name, dpi=150)
        plt.close(fig)

        visual_rows.append(
            {
                "item_idx": i,
                "debug_name": debug_name,
                "text": title_text,
                "image_id": image_id,
                "query_id": query_id,
                "file_name": file_name,
                "img_root": img_root,
                "original_path": os.path.join(img_root, file_name) if (img_root is not None and file_name is not None) else None,
                "prompt_box_source": prompt_box_source,
                "target_box_source": target_box_source,
                "prompt_box_raw": prompt_box_raw,
                "prompt_box_xyxy": prompt_box_xyxy,
                "target_box_raw": target_box_raw,
                "target_box_xyxy": target_box_xyxy,
                "json_raw_input_box": json_raw_input_box,
                "json_input_box_xyxy": json_box_xyxy,
                "json_first_ann_bbox": fallback_ann_bbox,
                "json_first_ann_bbox_xyxy": fallback_ann_xyxy,
                "gt_fg_ratio": float(gti.mean()),
                "pred_fg_ratio": float(pbi.mean()),
                "pred_prob_mean": float(probi.mean()),
            }
        )

    with open(save_dir / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(visual_rows, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] saved first-batch visuals -> {save_dir}")


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False  # PyTorch DDP default is False
    static_graph: bool = False  # PyTorch DDP default is False


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    wandb_writer: Optional[Any] = None




def _is_soft_prompt_param(name: str) -> bool:
    return name == "soft_prompt" or name.startswith("soft_prompt_proj.")


def _is_text_prompt_param(name: str) -> bool:
    if name == "sam3.backbone.language_backbone.encoder.text_projection":
        return True
    if ".ca_text." in name:
        return True
    if ".catext_norm." in name:
        return True
    return False


def _is_head_param(name: str) -> bool:
    if name.startswith("sam3.segmentation_head."):
        return True
    return False


def _set_lr_in_scheduler_cfg(scheduler_cfg: Dict[str, Any], lr_value: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(scheduler_cfg)
    if "scheduler" not in cfg or cfg["scheduler"] is None:
        cfg["value"] = float(lr_value)
        return cfg
    sch = cfg["scheduler"]
    if isinstance(sch, Mapping):
        if "base_lr" in sch:
            sch["base_lr"] = float(lr_value)
        elif "value" in sch:
            sch["value"] = float(lr_value)
        else:
            raise KeyError(f"Cannot set lr for scheduler config without 'base_lr' or 'value'. keys={list(sch.keys())}")
    else:
        raise TypeError(f"Unsupported scheduler config type: {type(sch)}")
    return cfg


def _collect_trainable_param_names_for_rs_tuning(model: torch.nn.Module):
    soft_names = []
    text_names = []
    head_names = []
    unexpected_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_soft_prompt_param(name):
            soft_names.append(name)
        elif _is_text_prompt_param(name):
            text_names.append(name)
        elif _is_head_param(name):
            head_names.append(name)
        else:
            unexpected_names.append(name)
    return soft_names, text_names, head_names, unexpected_names

class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else OptimConf()
        self.meters_conf = meters
        self.loss_conf = loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self.skip_first_val = skip_first_val
        self.skip_saving_ckpts = skip_saving_ckpts
        self.empty_gpu_mem_cache_after_eval = empty_gpu_mem_cache_after_eval
        self.force_debug_visual_once = os.getenv("SAM3_DEBUG_FORCE_FIRST_BATCH", "0") == "1"

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.  
        n_total = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"[AFTER FREEZE] total={n_total}, "
            f"trainable={n_trainable}, "
            f"frozen={n_total - n_trainable}"
        )

        debug_dir = os.path.join(self.logging_conf.log_dir, "debug_first_batch")
        dump_trainable_params(self.model, debug_dir)

        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        # 单卡/单进程时不包 DDP，避免:
        # 1) custom forward output (SAM3Output) 导致 DDP unused param tracing 失败
        # 2) semantic-only loss 只消费部分输出时的 reduction 报错
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world_size == 1:
            logging.info("World size == 1, skipping DistributedDataParallel wrapping.")
            return

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            static_graph=distributed_conf.static_graph,
        )

        if distributed_conf.comms_dtype is not None:
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        if self.skip_saving_ckpts:
            logging.info(
                "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
            )
            return
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        try:
            self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
            self.loss.load_state_dict(checkpoint["loss"], strict=True)
            self.epoch = checkpoint["epoch"]
            self.steps = checkpoint["steps"]
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

            if self.optim_conf.amp.enabled and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])

            self.best_meter_values = checkpoint.get("best_meter_values", {})

            if "train_dataset" in checkpoint and self.train_dataset is not None:
                self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])
        except ValueError as e:
            # Common when resuming from a checkpoint trained with a different
            # trainable-parameter set / optimizer param-group layout.
            msg = str(e)
            if "parameter group" in msg and "optimizer" in msg:
                logging.warning(
                    "Optimizer state is incompatible with current trainable parameter groups. "
                    "Falling back to model-only resume from checkpoint: %s",
                    ckpt_path,
                )
                self.epoch = 0
                self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}
                self.ckpt_time_elapsed = 0.0
                self.best_meter_values = {}
            else:
                raise

    def is_intermediate_val_epoch(self, epoch):
        skip_epoch = self.skip_first_val and epoch == 0
        return (
            epoch % self.val_epoch_freq == 0
            and epoch < self.max_epochs - 1
            and not skip_epoch
        )

    def _find_loss(self, key: str):
        if key in self.loss:
            return self.loss[key]

        assert key != "all", "Loss must be specified for key='all'"
        assert (
            "default" in self.loss
        ), f"Key {key} not found in losss, and no default provided"
        return self.loss["default"]

    def _find_meter(self, phase: str, key: str):
        if key in self.meters[phase]:
            return self.meters[phase][key]

        for cand_key, meter in self.meters[phase].items():
            if fnmatch.fnmatch(key, cand_key):
                return meter
        return None

    def _step(
        self,
        batch: BatchedDatapoint,
        model: nn.Module,
        phase: str,
    ):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        find_stages = model(batch)
        find_targets = [
            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets
        ]

        texts = _try_get_texts(batch)
        img_ids = _try_get_original_image_ids(batch)
        cat_ids = _try_get_original_category_ids(batch)

        if _debug_rank0_enabled(self.distributed_rank) and self.steps[phase] < 5:
            t0 = find_targets[0]
            raw_sem = t0["semantic_masks"] if "semantic_masks" in t0 else t0["masks"]
            raw_sem = raw_sem.float()

            _debug_print("[DEBUG][_step][RAW TARGET BEFORE LOSS]", rank=self.distributed_rank)
            _debug_print(
                f"[DEBUG] phase={phase} epoch={int(self.epoch)} step={int(self.steps[phase])}",
                rank=self.distributed_rank,
            )
            _debug_print("[DEBUG] raw_sem shape =", tuple(raw_sem.shape), rank=self.distributed_rank)
            _debug_print("[DEBUG] raw_sem batch_fg_ratio =", raw_sem.mean().item(), rank=self.distributed_rank)
            _debug_print(
                "[DEBUG] raw_sem per_sample_fg_ratio[:8] =",
                _fg_ratio_per_sample(raw_sem)[:8].tolist(),
                rank=self.distributed_rank,
            )
            if "masks" in t0:
                m = t0["masks"].float()
                _debug_print("[DEBUG] raw masks batch_fg_ratio =", m.mean().item(), rank=self.distributed_rank)
                _debug_print(
                    "[DEBUG] raw masks per_sample_fg_ratio[:8] =",
                    _fg_ratio_per_sample(m)[:8].tolist(),
                    rank=self.distributed_rank,
                )
            if "num_boxes" in t0 and torch.is_tensor(t0["num_boxes"]):
                _debug_print(
                    "[DEBUG] raw num_boxes[:8] =",
                    _to_cpu(t0["num_boxes"]).view(-1)[:8].tolist(),
                    rank=self.distributed_rank,
                )
            if img_ids is not None:
                _debug_print("[DEBUG] image_ids[:8] =", img_ids[:8].tolist(), rank=self.distributed_rank)
            if cat_ids is not None:
                _debug_print("[DEBUG] category_ids[:8] =", cat_ids[:8].tolist(), rank=self.distributed_rank)

        need_debug_dump = (
            self.distributed_rank == 0
            and (self.steps[phase] == 0 or self.force_debug_visual_once)
        )

        if need_debug_dump:
            debug_tag = f"{phase.lower()}_epoch{int(self.epoch):03d}_step{int(self.steps[phase]):06d}"
            debug_dir = os.path.join(self.logging_conf.log_dir, "debug_first_batch", debug_tag)
            debug_dataset = _get_dataset_for_phase(self, phase)
            ann_file, img_root = _discover_ann_file_and_img_root(debug_dataset)
            save_first_batch_visuals(
                batch,
                find_stages,
                find_targets,
                debug_dir,
                max_items=8,
                ann_file=ann_file,
                img_root=img_root,
            )
            self.force_debug_visual_once = False

        batch_size = len(batch.img_batch)
        loss_obj = self._find_loss(key)
        if hasattr(loss_obj, "__dict__"):
            loss_obj._debug_context = {
                "phase": phase,
                "epoch": int(self.epoch),
                "step": int(self.steps[phase]),
                "texts": texts[:8] if texts is not None else None,
                "image_ids": img_ids[:8].tolist() if img_ids is not None else None,
                "category_ids": cat_ids[:8].tolist() if cat_ids is not None else None,
            }
        loss = loss_obj(find_stages, find_targets)

        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase not in self.meters:
            return ret_tuple

        meters_dict = self._find_meter(phase, key)
        if meters_dict is None:
            return ret_tuple
        if meters_dict is not None:
            for _, meter in meters_dict.items():
                meter.update(
                    find_stages=find_stages,
                    find_metadatas=batch.find_metadatas,
                    model=model,
                    batch=batch,
                    key=key,
                )
            # Cleanup memory
            if isinstance(find_stages, SAM3Output):
                for fs in find_stages:
                    for k in list(fs.keys()):
                        del fs[k]

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()
                if torch.cuda.is_available() and self.empty_gpu_mem_cache_after_eval:
                    # release memory buffers held by the model during eval (which typically
                    # involves a lot more frames in video grounding that during training)
                    torch.cuda.empty_cache()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                # `default` is a fallback loss selector key, not a dataset key.
                # Logging it as a standalone meter is misleading (typically always zero).
                if key == "default":
                    continue
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # batch = batch.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        if loss_key in loss_mts:
                            loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            # `default` is a fallback loss selector key, not a dataset key.
            # Logging it as a standalone meter is misleading (typically always zero).
            if batch_key == "default":
                continue
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            # batch = batch.to(
            #     self.device, non_blocking=True
            # )  # move tensors in a tensorclass

            batch_for_eval = None
            if self.distributed_rank == 0 and _debug_enabled() and int(self.steps[phase]) < 5:
                try:
                    batch_for_eval = copy.deepcopy(batch)
                except Exception as e:
                    print(f"[DEBUG][post-step eval] deepcopy failed: {e}")
                    batch_for_eval = None

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                self._debug_post_step_eval_rerun(
                    batch=batch_for_eval if batch_for_eval is not None else batch,
                    phase=phase,
                    step_idx=max(int(self.steps[phase]) - 1, 0),
                    max_debug_steps=5,
                )

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)

        if self.gradient_accumulation_steps > 1:
            assert isinstance(
                batch, list
            ), f"Expected a list of batches, got {type(batch)}"
            assert (
                len(batch) == self.gradient_accumulation_steps
            ), f"Expected {self.gradient_accumulation_steps} batches, got {len(batch)}"
            accum_steps = len(batch)
        else:
            accum_steps = 1
            batch = [batch]

        for i, chunked_batch in enumerate(batch):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )
            with ddp_context:
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        chunked_batch,
                        self.model,
                        phase,
                    )

                assert len(loss_dict) == 1
                loss_key, loss = loss_dict.popitem()

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    if raise_on_error:
                        raise FloatingPointError(error_msg)
                    else:
                        return

                self.scaler.scale(loss).backward()
                loss_mts[loss_key].update(loss.item(), batch_size)
                for extra_loss_key, extra_loss in extra_losses.items():
                    if extra_loss_key not in extra_loss_mts:
                        extra_loss_mts[extra_loss_key] = AverageMeter(
                            extra_loss_key, self.device, ":.2e"
                        )
                    extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _debug_post_step_eval_rerun(
        self,
        batch,
        phase: str,
        step_idx: int,
        max_debug_steps: int = 5,
    ):
        if self.distributed_rank != 0:
            return
        if phase != Phase.TRAIN:
            return
        if step_idx >= max_debug_steps:
            return
        if not _debug_enabled():
            return
        if batch is None:
            print("[DEBUG][post-step eval] skip because batch is None")
            return

        if isinstance(batch, list):
            if len(batch) == 0:
                print("[DEBUG][post-step eval] skip empty batch list")
                return
            batch = batch[0]

        raw_batch = None
        if isinstance(batch, dict):
            if len(batch) == 0:
                print("[DEBUG][post-step eval] skip dict with unexpected len=0")
                return
            if len(batch) == 1:
                _, raw_batch = next(iter(batch.items()))
            elif "img_batch" in batch and "find_targets" in batch:
                raw_batch = batch
            else:
                print(f"[DEBUG][post-step eval] skip dict with unexpected len={len(batch)}")
                return
        elif hasattr(batch, "img_batch") and hasattr(batch, "find_targets"):
            raw_batch = batch
        else:
            print(f"[DEBUG][post-step eval] skip unsupported batch type: {type(batch)}")
            return

        texts = _try_get_texts(raw_batch)
        img_ids = _try_get_original_image_ids(raw_batch)
        cat_ids = _try_get_original_category_ids(raw_batch)

        model = self.model
        prev_mode = model.training
        model.eval()

        try:
            with torch.no_grad():
                device_type = self.device.type if hasattr(self.device, "type") else "cuda"
                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    eval_batch = copy_data_to_device(raw_batch, self.device, non_blocking=True)

                    eval_find_stages = model(eval_batch)
                    eval_find_targets = [
                        unwrap_ddp_if_wrapped(model).back_convert(x)
                        for x in eval_batch.find_targets
                    ]

                    eval_out = _unwrap_stage_output(eval_find_stages)
                    if "semantic_seg" not in eval_out:
                        print("[DEBUG][post-step eval] semantic_seg not found")
                        return

                    pred = eval_out["semantic_seg"].float()
                    pred_prob_288 = pred.sigmoid()
                    pred_bin_288 = pred_prob_288 > 0.5

                    t0 = eval_find_targets[0]
                    semantic_targets_1008 = (
                        t0["semantic_masks"] if "semantic_masks" in t0 else t0["masks"]
                    ).bool()

                    pred_up_1008 = F.interpolate(
                        pred,
                        size=semantic_targets_1008.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_prob_1008 = pred_up_1008.sigmoid()
                    pred_bin_1008 = pred_prob_1008 > 0.5

                    semantic_targets_288 = (
                        F.interpolate(
                            semantic_targets_1008.float().unsqueeze(1),
                            size=pred.shape[-2:],
                            mode="nearest",
                        )
                        .squeeze(1)
                        .bool()
                    )

                    inter_288 = (pred_bin_288[:, 0] & semantic_targets_288).flatten(1).sum(1).float()
                    union_288 = (pred_bin_288[:, 0] | semantic_targets_288).flatten(1).sum(1).float()
                    miou_288 = (inter_288 / torch.clamp(union_288, min=1.0)).mean()

                    inter_1008 = (pred_bin_1008[:, 0] & semantic_targets_1008).flatten(1).sum(1).float()
                    union_1008 = (pred_bin_1008[:, 0] | semantic_targets_1008).flatten(1).sum(1).float()
                    miou_1008 = (inter_1008 / torch.clamp(union_1008, min=1.0)).mean()

                    print("[DEBUG][post-step eval rerun]")
                    print(f"[DEBUG] phase={phase} epoch={int(self.epoch)} step={int(step_idx)}")
                    if texts is not None:
                        print("[DEBUG] texts[:8] =", texts[:8])
                    if img_ids is not None:
                        print("[DEBUG] image_ids[:8] =", img_ids[:8].tolist())
                    if cat_ids is not None:
                        print("[DEBUG] category_ids[:8] =", cat_ids[:8].tolist())
                    print(
                        f"[DEBUG] pred_288 prob_mean={pred_prob_288.mean().item():.6f} "
                        f"pred_fg@0.5={pred_bin_288.float().mean().item():.6f} "
                        f"target_fg={semantic_targets_288.float().mean().item():.6f} "
                        f"miou_288={miou_288.item():.6f}"
                    )
                    print(
                        f"[DEBUG] pred_1008 prob_mean={pred_prob_1008.mean().item():.6f} "
                        f"pred_fg@0.5={pred_bin_1008.float().mean().item():.6f} "
                        f"target_fg={semantic_targets_1008.float().mean().item():.6f} "
                        f"miou_1008={miou_1008.item():.6f}"
                    )
        finally:
            model.train(prev_mode)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                if "default" not in loss_keys:
                    for k in val_keys:
                        assert (
                            k in loss_keys
                        ), f"Error: key {k} is not defined in the losses, and no default is set"

    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        maybe_apply_remote_sensing_freeze(self.model)

        def _minimal_unfreeze_for_rs(model, preset: str = None):
            if preset is None:
                preset = os.getenv("SAM3_TEXT_UNFREEZE_PRESET", "text_last1")
            preset = str(preset).strip().lower().replace("-", "_").replace("+", "_")

            for _, p in model.named_parameters():
                p.requires_grad = False

            soft_exact_names = {
                n for n, _ in model.named_parameters() if _is_soft_prompt_param(n)
            }
            if len(soft_exact_names) == 0:
                soft_exact_names = {"soft_prompt"}
            exact_names = set(soft_exact_names)
            text_proj_name = "sam3.backbone.language_backbone.encoder.text_projection"

            layer_prefixes = {
                "l0": (
                    "sam3.transformer.decoder.layers.0.ca_text.",
                    "sam3.transformer.decoder.layers.0.catext_norm.",
                ),
                "l1": (
                    "sam3.transformer.decoder.layers.1.ca_text.",
                    "sam3.transformer.decoder.layers.1.catext_norm.",
                ),
                "l2": (
                    "sam3.transformer.decoder.layers.2.ca_text.",
                    "sam3.transformer.decoder.layers.2.catext_norm.",
                ),
                "l3": (
                    "sam3.transformer.decoder.layers.3.ca_text.",
                    "sam3.transformer.decoder.layers.3.catext_norm.",
                ),
                "l4": (
                    "sam3.transformer.decoder.layers.4.ca_text.",
                    "sam3.transformer.decoder.layers.4.catext_norm.",
                ),
                "l5": (
                    "sam3.transformer.decoder.layers.5.ca_text.",
                    "sam3.transformer.decoder.layers.5.catext_norm.",
                ),
            }

            prefixes = ()
            if preset == "soft_only":
                exact_names = set(soft_exact_names)
            elif preset == "text_proj_only":
                exact_names = set(soft_exact_names) | {text_proj_name}
            elif preset == "text_last1":
                exact_names = set(soft_exact_names) | {text_proj_name}
                prefixes = layer_prefixes["l5"]
            elif preset == "text_last2":
                exact_names = set(soft_exact_names) | {text_proj_name}
                prefixes = layer_prefixes["l4"] + layer_prefixes["l5"]
            elif preset == "text_all6":
                exact_names = set(soft_exact_names) | {text_proj_name}
                prefixes = (
                    layer_prefixes["l0"]
                    + layer_prefixes["l1"]
                    + layer_prefixes["l2"]
                    + layer_prefixes["l3"]
                    + layer_prefixes["l4"]
                    + layer_prefixes["l5"]
                )
            elif preset == "head_prompt":
                exact_names = set(soft_exact_names)
                prefixes = ("sam3.segmentation_head.",)
            else:
                supported = [
                    "soft_only",
                    "text_proj_only",
                    "text_last1",
                    "text_last2",
                    "text_all6",
                    "head_prompt",
                ]
                raise ValueError(
                    f"Unknown preset: {preset}. Supported presets: {supported}"
                )

            changed = []
            trainable_numel = 0
            for name, p in model.named_parameters():
                keep = (name in exact_names) or any(name.startswith(pr) for pr in prefixes)
                if keep:
                    p.requires_grad = True
                    changed.append(name)
                    trainable_numel += p.numel()

            print(f"[TEXT-ONLY UNFREEZE] preset={preset}")
            print(f"[TEXT-ONLY UNFREEZE] num_params={trainable_numel}")
            print(f"[TEXT-ONLY UNFREEZE] num_tensors={len(changed)}")
            for n in changed:
                print("   ", n)

        _minimal_unfreeze_for_rs(self.model)
        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        # Safer default for soft-prompt tuning to reduce prompt drift / over-activation.
        soft_lr = float(os.getenv("SAM3_SOFT_PROMPT_LR", "5e-5"))
        text_lr = float(os.getenv("SAM3_TEXT_LR", "1e-5"))
        head_lr = float(os.getenv("SAM3_HEAD_LR", str(soft_lr)))

        if self.optim_conf.options is None:
            raise ValueError("self.optim_conf.options is None; cannot build optimizer param groups.")

        options_conf = copy.deepcopy(self.optim_conf.options)
        if "lr" not in options_conf or options_conf["lr"] is None or len(options_conf["lr"]) == 0:
            raise KeyError("Optimizer options must contain a non-empty 'lr' config list.")

        soft_names, text_names, head_names, unexpected_names = _collect_trainable_param_names_for_rs_tuning(self.model)
        if len(soft_names) == 0:
            raise RuntimeError("No trainable 'soft_prompt' parameter found.")
        if len(text_names) == 0 and len(head_names) == 0:
            logging.warning(
                "[TEXT-ONLY OPTIM] No trainable text/head-layer params matched; optimizer becomes soft-prompt-only."
            )
        if len(unexpected_names) > 0:
            raise RuntimeError(
                "[TEXT-ONLY OPTIM] Found trainable params outside soft/text/head groups:\n"
                + "\n".join(unexpected_names[:200])
            )

        base_lr_cfg = copy.deepcopy(options_conf["lr"][0])
        soft_lr_cfg = _set_lr_in_scheduler_cfg(base_lr_cfg, soft_lr)
        soft_lr_cfg["param_names"] = sorted(soft_names)

        if len(text_names) > 0:
            text_lr_cfg = _set_lr_in_scheduler_cfg(base_lr_cfg, text_lr)
            text_lr_cfg["param_names"] = sorted(text_names)
            options_conf["lr"] = [soft_lr_cfg, text_lr_cfg]
        else:
            options_conf["lr"] = [soft_lr_cfg]

        if len(head_names) > 0:
            head_lr_cfg = _set_lr_in_scheduler_cfg(base_lr_cfg, head_lr)
            head_lr_cfg["param_names"] = sorted(head_names)
            options_conf["lr"].append(head_lr_cfg)

        param_allowlist = set(soft_names) | set(text_names) | set(head_names)
        logging.info("[TEXT-ONLY OPTIM] soft_prompt lr = %.8f", soft_lr)
        logging.info("[TEXT-ONLY OPTIM] text-layer  lr = %.8f", text_lr)
        logging.info("[TEXT-ONLY OPTIM] head-layer  lr = %.8f", head_lr)
        logging.info("[TEXT-ONLY OPTIM] #soft tensors = %d", len(soft_names))
        logging.info("[TEXT-ONLY OPTIM] #text tensors = %d", len(text_names))
        logging.info("[TEXT-ONLY OPTIM] #head tensors = %d", len(head_names))

        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            options_conf,
            self.optim_conf.param_group_modifiers,
            param_allowlist=param_allowlist,
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
