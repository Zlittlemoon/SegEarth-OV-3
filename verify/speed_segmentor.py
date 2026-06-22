"""
Measure end-to-end per-image inference speed of SegEarthOV3Segmentation in
serial vs batch grounding mode, using the real three-head fusion segmentor.

Two timings are reported for each mode:

  full   : set_image + grounding (what the eval loop actually pays per image)
  grounding-only : grounding alone, image pre-encoded once and reused
                   (isolates the part batch mode changes)

Run:
    python verify/speed_segmentor.py
    python verify/speed_segmentor.py --classes configs/cls_loveda.txt --image resources/loveda_2524.png
    python verify/speed_segmentor.py --repeat 30 --warmup 5
"""

import argparse
import time

import torch
from PIL import Image

import segearthov3_segmentor_merge as seg_mod
from segearthov3_segmentor_merge import SegEarthOV3Segmentation


def build_model(classname_path, mode, confidence_threshold=0.5):
    model = SegEarthOV3Segmentation(
        classname_path=classname_path,
        confidence_threshold=confidence_threshold,
        prob_thd=0.0,
        slide_stride=0,
        slide_crop=0,
        sam3_infer_mode=mode,
    )
    # eval() is essential: disables Dropout, switches BN to running stats
    model.processor.model.eval()
    return model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_loop(fn, repeat):
    sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeat * 1e3  # ms


def measure(model, image, repeat, warmup):
    """Return (full_ms, grounding_ms) for the given model."""

    # ---- full: set_image + grounding (what the eval loop pays per image) ----
    def full_once():
        model._inference_single_view(image)

    # ---- grounding-only: image pre-encoded once, reused across timed calls ----
    state_holder = {}

    @torch.inference_mode()
    def prepare_state():
        state_holder['state'] = model.processor.set_image(image)

    prepare_state()
    state = state_holder['state']

    if model.sam3_infer_mode == 'batch':
        from sam3.model.data_misc import FindStage
        
        @torch.no_grad()
        def grounding_once():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                n = model.num_queries
                text_outputs = model.processor.model.backbone.forward_text(
                    model.query_words, device=model.device)
                state["backbone_out"].update(text_outputs)
                geometric_prompt = model.processor.model._get_dummy_prompt(num_prompts=n)
                find_stage = FindStage(
                    img_ids=torch.zeros(n, device=model.device, dtype=torch.long),
                    text_ids=torch.arange(n, device=model.device, dtype=torch.long),
                    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                    input_points=None, input_points_mask=None,
                )
                model.processor.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )
    else:
        @torch.no_grad()
        def grounding_once():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for query_word in model.query_words:
                    model.processor.reset_all_prompts(state)
                    model.processor.set_text_prompt(state=state, prompt=query_word)

    # warmup
    for _ in range(warmup):
        full_once()
        grounding_once()

    full_ms = time_loop(full_once, repeat)
    grounding_ms = time_loop(grounding_once, repeat)
    return full_ms, grounding_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="resources/loveda_2524.png")
    ap.add_argument("--classes", default="configs/cls_vdd.txt")
    ap.add_argument("--confidence_threshold", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    n_cls = sum(1 for line in open(args.classes) if line.strip())
    print(f"Image : {args.image}  ({image.size[0]}x{image.size[1]})")
    print(f"Classes: {args.classes}  ({n_cls} lines)")
    print(f"Repeat : {args.repeat}  Warmup: {args.warmup}\n")

    print("Building serial model ...")
    serial_model = build_model(args.classes, 'serial', args.confidence_threshold)
    print("Building batch model  ...")
    batch_model  = build_model(args.classes, 'batch',  args.confidence_threshold)

    print("\n--- Serial mode ---")
    s_full, s_grnd = measure(serial_model, image, args.repeat, args.warmup)
    print(f"  full (set_image + grounding) : {s_full:8.2f} ms")
    print(f"  grounding-only               : {s_grnd:8.2f} ms")

    print("\n--- Batch mode  ---")
    b_full, b_grnd = measure(batch_model, image, args.repeat, args.warmup)
    print(f"  full (set_image + grounding) : {b_full:8.2f} ms")
    print(f"  grounding-only               : {b_grnd:8.2f} ms")

    print("\n--- Speedup ---")
    print(f"  full       speedup : {s_full/b_full:.2f}x")
    print(f"  grounding  speedup : {s_grnd/b_grnd:.2f}x")
    set_img_ms = s_full - s_grnd
    print(f"  (set_image cost ~{set_img_ms:.1f} ms — shared fixed cost, not affected by mode)")


if __name__ == "__main__":
    main()
