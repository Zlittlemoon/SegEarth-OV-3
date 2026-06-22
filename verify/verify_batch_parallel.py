import argparse
import time
from contextlib import nullcontext

import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage
from segearthov3_segmentor_original import get_cls_idx


# Keys of the raw forward_grounding output we compare, grouped by head.
HEAD_KEYS = {
    "semantic": ["semantic_seg"],
    "presence": ["presence_logit_dec"],
    "instance": ["pred_masks", "pred_logits"],
}


def load_classes(path):

    return get_cls_idx(path)  # (query_words, query_idx)

def build_find_stage(n, device):
    """All prompts look at image 0; each prompt indexes its own text feature."""
    return FindStage(
        img_ids=torch.zeros(n, device=device, dtype=torch.long),
        text_ids=torch.arange(n, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


@torch.inference_mode()
def run_grounding(processor, state, prompts):
    """Run a single forward_grounding for a list of prompts (batch size = len)."""
    n = len(prompts)
    text_outputs = processor.model.backbone.forward_text(prompts, device=processor.device)
    # update() overwrites any previous text features on the shared backbone_out
    state["backbone_out"].update(text_outputs)
    geometric_prompt = processor.model._get_dummy_prompt(num_prompts=n)
    find_stage = build_find_stage(n, processor.device)
    return processor.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )


def _collect(out, keys):
    return {k: out[k].detach().float().cpu() for k in keys if k in out}

def _flatten_tensors(obj, prefix=""):
    """
    Recursively collect all tensors from nested dict/list/tuple outputs.

    Example:
        {"a": Tensor, "b": {"c": Tensor}}
    becomes:
        {"a": Tensor, "b.c": Tensor}
    """
    items = {}

    if torch.is_tensor(obj):
        key = prefix if prefix else "tensor"
        items[key] = obj.detach().float().cpu()
        return items

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            items.update(_flatten_tensors(v, new_prefix))
        return items

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            items.update(_flatten_tensors(v, new_prefix))
        return items

    return items


@torch.inference_mode()
def serial_text_outputs(processor, prompts, autocast):
    """
    Run forward_text one prompt at a time and combine outputs
    into the same layout as forward_text(prompts).

    Expected SAM3 text output layout:
        language_embeds   : [T, B, 1024]
        language_features : [T, B, 256]
        language_mask     : [B, T]
    """
    per_prompt = []

    with autocast:
        for p in prompts:
            out = processor.model.backbone.forward_text(
                [p],
                device=processor.device
            )
            per_prompt.append(_flatten_tensors(out))

    common_keys = set(per_prompt[0].keys())
    for d in per_prompt[1:]:
        common_keys &= set(d.keys())

    stacked = {}

    for k in sorted(common_keys):
        vals = [d[k] for d in per_prompt]
        v0 = vals[0]

        # language_embeds / language_features:
        # single prompt: [T, 1, D]
        # multi prompt : [T, N, D]
        if v0.ndim == 3 and v0.shape[1] == 1:
            stacked[k] = torch.cat(vals, dim=1)

        # language_mask:
        # single prompt: [1, T]
        # multi prompt : [N, T]
        elif v0.ndim == 2 and v0.shape[0] == 1:
            stacked[k] = torch.cat(vals, dim=0)

        # fallback
        else:
            stacked[k] = torch.stack(vals, dim=0)

    return stacked


@torch.inference_mode()
def batch_text_outputs(processor, prompts, autocast):
    """
    Run forward_text once with all prompts.
    """
    with autocast:
        out = processor.model.backbone.forward_text(
            prompts,
            device=processor.device
        )

    return _flatten_tensors(out)

def _merge_serial_text_values(vals):
    """
    Merge a list of forward_text([prompt]) outputs into the same layout as
    forward_text([prompt1, prompt2, ...]).

    Expected main SAM3 text layouts:
        language_embeds   : [T, 1, D] -> [T, N, D]
        language_features : [T, 1, D] -> [T, N, D]
        language_mask     : [1, T]    -> [N, T]
    """
    v0 = vals[0]

    if torch.is_tensor(v0):
        # language_embeds / language_features: [T, 1, D] -> [T, N, D]
        if v0.ndim == 3 and v0.shape[1] == 1:
            return torch.cat(vals, dim=1)

        # language_mask: [1, T] -> [N, T]
        if v0.ndim == 2 and v0.shape[0] == 1:
            return torch.cat(vals, dim=0)

        # fallback
        return torch.stack(vals, dim=0)

    if isinstance(v0, dict):
        return {
            k: _merge_serial_text_values([v[k] for v in vals])
            for k in v0.keys()
        }

    if isinstance(v0, (list, tuple)):
        merged = [
            _merge_serial_text_values([v[i] for v in vals])
            for i in range(len(v0))
        ]
        return type(v0)(merged)

    # non-tensor metadata, usually shared
    return v0


@torch.inference_mode()
def serial_text_outputs_raw(processor, prompts, autocast):
    """
    Run forward_text([p]) one prompt at a time, but keep tensors on device
    and preserve the original dict structure.

    This is different from serial_text_outputs(), which flattens and moves
    tensors to CPU only for comparison.
    """
    outs = []

    with autocast:
        for p in prompts:
            out = processor.model.backbone.forward_text(
                [p],
                device=processor.device
            )
            outs.append(out)

    return _merge_serial_text_values(outs)


def compare_text_encoder(processor, prompts, autocast, atol, rtol):
    print("== Text encoder equivalence check ==")

    serial = serial_text_outputs(processor, prompts, autocast)
    batch = batch_text_outputs(processor, prompts, autocast)

    serial_keys = set(serial.keys())
    batch_keys = set(batch.keys())

    only_serial = sorted(serial_keys - batch_keys)
    only_batch = sorted(batch_keys - serial_keys)
    common = sorted(serial_keys & batch_keys)

    if only_serial:
        print(f"Only in serial text output: {only_serial}")
    if only_batch:
        print(f"Only in batch text output: {only_batch}")

    ok_all = True

    for k in common:
        a, b = serial[k], batch[k]

        if a.shape != b.shape:
            print(
                f"  [{k}] SHAPE MISMATCH "
                f"serial={tuple(a.shape)} batch={tuple(b.shape)}"
            )
            ok_all = False
            continue

        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = a.abs().max().item() + 1e-12
        rel = max_abs / denom

        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok_all = ok_all and ok

        print(
            f"  [{k}] shape={tuple(a.shape)} "
            f"mean|diff|={mean_abs:.3e} "
            f"max|diff|={max_abs:.3e} "
            f"rel={rel:.3e} "
            f"-> {'OK' if ok else 'DIFF'}"
        )

    print(f"\nText encoder result: {'ALL MATCH' if ok_all else 'MISMATCH'}")
    return ok_all  


@torch.inference_mode()
def serial_outputs(processor, image, prompts, keys, autocast):
    """Loop the grounding one prompt at a time, stack results along the batch dim."""
    state = processor.set_image(image)
    per_prompt = []
    with autocast:
        for p in prompts:
            out = run_grounding(processor, state, [p])
            per_prompt.append(_collect(out, keys))
    # stack the (single-prompt) tensors back into (N, ...)
    stacked = {}
    for k in keys:
        if k in per_prompt[0]:
            stacked[k] = torch.cat([d[k] for d in per_prompt], dim=0)
    return stacked


@torch.inference_mode()
def batch_outputs(processor, image, prompts, keys, autocast):
    """One grounding call with all prompts batched together."""
    state = processor.set_image(image)
    with autocast:
        out = run_grounding(processor, state, prompts)
    return _collect(out, keys)

@torch.inference_mode()
def batch_outputs_with_serial_text(processor, image, prompts, keys, autocast):
    """
    Isolation experiment:

        serial forward_text([p]) for each prompt
        + one batch forward_grounding()

    This fixes the text features to the serial version, and only changes
    the grounding stage from serial to batch.
    """
    state = processor.set_image(image)
    n = len(prompts)

    # 1. Build text features using serial forward_text([p])
    text_outputs = serial_text_outputs_raw(
        processor=processor,
        prompts=prompts,
        autocast=autocast,
    )

    # 2. Replace the text features in the shared image state
    state["backbone_out"].update(text_outputs)

    # 3. Run one batch grounding with img_ids=[0,...,0], text_ids=[0,...,N-1]
    geometric_prompt = processor.model._get_dummy_prompt(num_prompts=n)
    find_stage = build_find_stage(n, processor.device)

    with autocast:
        out = processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    return _collect(out, keys)


@torch.inference_mode()
def batch_outputs_repeated_images(processor, image, prompts, keys, autocast):
    """
    Diagnostic experiment:

        set_image_batch([image] * N)
        img_ids  = [0, 1, 2, ..., N-1]
        text_ids = [0, 1, 2, ..., N-1]

    This checks whether using distinct image ids makes batch grounding closer
    to serial inference.
    """
    n = len(prompts)

    # Encode the same PIL image as an image batch of size N.
    state = processor.set_image_batch([image] * n)

    # Keep text features equal to the serial-text version.
    text_outputs = serial_text_outputs_raw(
        processor=processor,
        prompts=prompts,
        autocast=autocast,
    )
    state["backbone_out"].update(text_outputs)

    geometric_prompt = processor.model._get_dummy_prompt(num_prompts=n)

    # The difference from build_find_stage():
    #   build_find_stage: img_ids = [0, 0, ..., 0]
    #   this experiment : img_ids = [0, 1, ..., N-1]
    find_stage = FindStage(
        img_ids=torch.arange(n, device=processor.device, dtype=torch.long),
        text_ids=torch.arange(n, device=processor.device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    with autocast:
        out = processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    return _collect(out, keys)


def compare(serial, batch, atol, rtol):
    ok_all = True
    for k in serial:
        a, b = serial[k], batch[k]
        if a.shape != b.shape:
            print(f"  [{k}] SHAPE MISMATCH serial={tuple(a.shape)} batch={tuple(b.shape)}")
            ok_all = False
            continue
        max_abs = (a - b).abs().max().item()
        denom = a.abs().max().item() + 1e-12
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok_all = ok_all and ok
        print(
            f"  [{k}] shape={tuple(a.shape)} "
            f"max|abs diff|={max_abs:.3e} (rel {max_abs/denom:.3e}) "
            f"-> {'OK' if ok else 'DIFF'}"
        )
    return ok_all

def compare_semantic_argmax(serial, batch, name=""):
    """
    Compare final category prediction induced by semantic_seg.

    Expected semantic_seg shape:
        [C, 1, H, W]
    where C is the number of prompts/classes.
    """
    if "semantic_seg" not in serial or "semantic_seg" not in batch:
        return

    s = serial["semantic_seg"].squeeze(1)  # [C, H, W]
    b = batch["semantic_seg"].squeeze(1)   # [C, H, W]

    if s.ndim != 3 or b.ndim != 3 or s.shape[0] < 2:
        print(f"  [semantic argmax{name}] skipped: need at least 2 prompts")
        return

    if s.shape != b.shape:
        print(
            f"  [semantic argmax{name}] SHAPE MISMATCH "
            f"serial={tuple(s.shape)} batch={tuple(b.shape)}"
        )
        return

    s_pred = s.argmax(dim=0)
    b_pred = b.argmax(dim=0)

    changed = s_pred != b_pred
    agree = (~changed).float().mean().item()

    print(f"  [semantic argmax{name}] pixel agreement = {agree * 100:.4f}%")
    print(f"  [semantic argmax{name}] changed pixels  = {changed.float().mean().item() * 100:.4f}%")

    s_top2 = torch.topk(s, k=2, dim=0).values
    s_margin = s_top2[0] - s_top2[1]

    if changed.any():
        print(
            f"  [semantic argmax{name}] mean serial margin on changed pixels = "
            f"{s_margin[changed].mean().item():.6f}"
        )
    else:
        print(f"  [semantic argmax{name}] no changed pixels")


def compare_instance_top_query(serial, batch, name=""):
    """
    Compare whether the instance branch selects the same top query for each prompt.
    Also compare selected binary masks if pred_masks are available.
    """
    if "pred_logits" not in serial or "pred_logits" not in batch:
        return

    s_logit = serial["pred_logits"].squeeze(-1)  # [C, Q]
    b_logit = batch["pred_logits"].squeeze(-1)   # [C, Q]

    if s_logit.shape != b_logit.shape:
        print(
            f"  [instance top-query{name}] SHAPE MISMATCH "
            f"serial={tuple(s_logit.shape)} batch={tuple(b_logit.shape)}"
        )
        return

    s_top = s_logit.argmax(dim=1)
    b_top = b_logit.argmax(dim=1)

    agree = (s_top == b_top).float().mean().item()

    print(f"  [instance top-query{name}] agreement = {agree * 100:.2f}%")
    print(f"  [instance top-query{name}] serial top ids = {s_top.tolist()}")
    print(f"  [instance top-query{name}] batch  top ids = {b_top.tolist()}")

    if "pred_masks" in serial and "pred_masks" in batch:
        if serial["pred_masks"].shape != batch["pred_masks"].shape:
            print(
                f"  [instance selected-mask{name}] SHAPE MISMATCH "
                f"serial={tuple(serial['pred_masks'].shape)} "
                f"batch={tuple(batch['pred_masks'].shape)}"
            )
            return

        c = s_logit.shape[0]
        batch_idx = torch.arange(c)

        s_mask = serial["pred_masks"][batch_idx, s_top] > 0
        b_mask = batch["pred_masks"][batch_idx, b_top] > 0

        mask_agree = (s_mask == b_mask).float().mean().item()
        print(f"  [instance selected-mask{name}] binary pixel agreement = {mask_agree * 100:.4f}%")


def speed_test(processor, image, prompts, autocast, repeat, warmup):
    cuda = torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def time_loop(fn, n):
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) / n

    # ---- end-to-end: set_image is timed (fixed backbone cost included) ----
    @torch.inference_mode()
    def serial_e2e():
        state = processor.set_image(image)
        with autocast:
            for p in prompts:
                run_grounding(processor, state, [p])

    @torch.inference_mode()
    def batch_e2e():
        state = processor.set_image(image)
        with autocast:
            run_grounding(processor, state, prompts)
        
    # ---- grounding-only: image encoded once, reused across timed calls ----
    state = processor.set_image(image)  # the shared, pre-encoded image    
    
    @torch.inference_mode()
    def serial_grounding():
        with autocast:
            for p in prompts:
                run_grounding(processor, state, [p])
   
    @torch.inference_mode()
    def batch_grounding():
        with autocast:
            run_grounding(processor, state, prompts)

    for _ in range(warmup):
        serial_e2e(); batch_e2e()
        serial_grounding(); batch_grounding()

    t_serial_e2e = time_loop(serial_e2e, repeat)
    t_batch_e2e = time_loop(batch_e2e, repeat)
    t_serial_g = time_loop(serial_grounding, repeat)
    t_batch_g = time_loop(batch_grounding, repeat)

    print(f"\nSpeed over {repeat} runs ({len(prompts)} prompts):")
    print("  -- end-to-end (incl. set_image / image backbone) --")
    print(f"    serial loop : {t_serial_e2e*1e3:8.2f} ms")
    print(f"    batch once  : {t_batch_e2e*1e3:8.2f} ms")
    print(f"    speedup     : {t_serial_e2e/t_batch_e2e:8.2f}x")
    print("  -- grounding-only (image pre-encoded, reused) --")
    print(f"    serial loop : {t_serial_g*1e3:8.2f} ms")
    print(f"    batch once  : {t_batch_g*1e3:8.2f} ms")
    print(f"    speedup     : {t_serial_g/t_batch_g:8.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/Potsdam/img_dir/val/2_13_5120_0_5632_512.png")
    ap.add_argument("--classes", default="configs/cls_potsdam.txt")
    ap.add_argument("--num", type=int, default=5, help="number of prompts (categories) to use")
    ap.add_argument(
        "--stage",
        choices=["text", "semantic", "presence", "instance", "all"],
        default="all"
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="float32 for a strict numerical check; bfloat16 matches the segmentor")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--checkpoint", default="weights/sam3/sam3.pt")
    ap.add_argument("--bpe", default="./sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    ap.add_argument("--speed", action="store_true", help="run the speed comparison")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
    "--isolate-text",
    action="store_true",
    help="Use serial-encoded text features but run batch forward_grounding."
    )
    ap.add_argument(
        "--repeat-image-batch",
        action="store_true",
        help="Use set_image_batch([image] * N) with img_ids=arange(N)."
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    # default tolerance: loose for bf16 (different reduction order), tight for fp32
    atol = args.atol if args.atol is not None else (5e-2 if dtype == torch.bfloat16 else 1e-4)
    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if (device == "cuda" and dtype == torch.bfloat16)
        else nullcontext()
    )

    query_words, query_idx = load_classes(args.classes)
    prompts = query_words[: args.num]
    print(f"Prompts ({len(prompts)}): {prompts}")
    print(f"(synonyms already split, matching SegEarth's get_cls_idx() exactly)")
    print(f"device={device} dtype={args.dtype} atol={atol} rtol={args.rtol}\n")

    model = build_sam3_image_model(
        bpe_path=args.bpe,
        checkpoint_path=args.checkpoint,
        device=device
    )
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.0, device=device)

    if args.stage == "text":
        compare_text_encoder(
            processor=processor,
            prompts=prompts,
            autocast=autocast,
            atol=atol,
            rtol=args.rtol,
        )
        return

    image = Image.open(args.image).convert("RGB")

    stages = ["semantic", "presence", "instance"] if args.stage == "all" else [args.stage]
    keys = [k for s in stages for k in HEAD_KEYS[s]]

    print("== Equivalence check (raw forward_grounding outputs) ==")

    # A. Serial Text + Serial Grounding
    serial = serial_outputs(processor, image, prompts, keys, autocast)

    # B. Batch Text + Batch Grounding
    batch = batch_outputs(processor, image, prompts, keys, autocast)

    print("\n== Compare A vs B: Serial Text + Serial Grounding  VS  Batch Text + Batch Grounding ==")
    overall = True
    for s in stages:
        print(f"-- {s} head --")
        sub_serial = {k: serial[k] for k in HEAD_KEYS[s] if k in serial}
        sub_batch = {k: batch[k] for k in HEAD_KEYS[s] if k in batch}
        overall = compare(sub_serial, sub_batch, atol, args.rtol) and overall

    print(f"\nNormal batch result: {'ALL MATCH' if overall else 'MISMATCH (see above)'}")
    print("\n== Prediction-level metrics: A vs B ==")
    compare_semantic_argmax(serial, batch, name=" A-vs-B")
    compare_instance_top_query(serial, batch, name=" A-vs-B")


    # C. Serial Text + Batch Grounding
    if args.isolate_text:
        batch_serial_text = batch_outputs_with_serial_text(
            processor=processor,
            image=image,
            prompts=prompts,
            keys=keys,
            autocast=autocast,
        )

        print("\n== Compare A vs C: Serial Text + Serial Grounding  VS  Serial Text + Batch Grounding ==")
        overall_iso = True
        for s in stages:
            print(f"-- {s} head --")
            sub_serial = {k: serial[k] for k in HEAD_KEYS[s] if k in serial}
            sub_batch_st = {k: batch_serial_text[k] for k in HEAD_KEYS[s] if k in batch_serial_text}
            overall_iso = compare(sub_serial, sub_batch_st, atol, args.rtol) and overall_iso

        print(f"\nSerial-text batch-grounding result: {'ALL MATCH' if overall_iso else 'MISMATCH (see above)'}")

        print("\n== Prediction-level metrics: A vs C ==")
        compare_semantic_argmax(serial, batch_serial_text, name=" A-vs-C")
        compare_instance_top_query(serial, batch_serial_text, name=" A-vs-C")

    # D. Repeated-Image Batch Grounding
    if args.repeat_image_batch:
        batch_repeat_img = batch_outputs_repeated_images(
            processor=processor,
            image=image,
            prompts=prompts,
            keys=keys,
            autocast=autocast,
        )

        print("\n== Compare A vs D: Serial Text + Serial Grounding  VS  Repeated-Image Batch Grounding ==")
        overall_rep = True
        for s in stages:
            print(f"-- {s} head --")
            sub_serial = {k: serial[k] for k in HEAD_KEYS[s] if k in serial}
            sub_rep = {k: batch_repeat_img[k] for k in HEAD_KEYS[s] if k in batch_repeat_img}
            overall_rep = compare(sub_serial, sub_rep, atol, args.rtol) and overall_rep

        print(f"\nRepeated-image batch result: {'ALL MATCH' if overall_rep else 'MISMATCH (see above)'}")

        print("\n== Prediction-level metrics: A vs D ==")
        compare_semantic_argmax(serial, batch_repeat_img, name=" A-vs-D")
        compare_instance_top_query(serial, batch_repeat_img, name=" A-vs-D")

    if args.speed:
        speed_test(processor, image, prompts, autocast, args.repeat, args.warmup)


if __name__ == "__main__":
    main()
