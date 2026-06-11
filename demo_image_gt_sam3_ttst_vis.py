import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mmseg.structures import SegDataSample
from torchvision import transforms

from segearthov3_segmentor_original import SegEarthOV3Segmentation
from segearthov3_segmentor_merge_sr_input import SegEarthOV3SegmentationSRInput


PALETTES = {
    "dlrsd": np.array(
        [
            [166, 202, 240],
            [128, 128, 0],
            [0, 0, 128],
            [255, 0, 0],
            [0, 128, 0],
            [128, 0, 0],
            [255, 233, 233],
            [160, 160, 164],
            [0, 128, 128],
            [90, 87, 255],
            [255, 255, 0],
            [255, 192, 0],
            [0, 0, 255],
            [255, 0, 192],
            [128, 0, 128],
            [0, 255, 0],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    ),
    "isaid_split": np.array(
        [
            [0, 0, 63],
            [0, 63, 63],
            [0, 63, 0],
            [0, 63, 127],
            [0, 63, 191],
            [0, 63, 255],
            [0, 127, 63],
            [0, 127, 127],
            [0, 0, 127],
            [0, 0, 191],
            [0, 0, 255],
            [0, 191, 127],
            [0, 127, 191],
            [0, 127, 255],
            [0, 100, 155],
        ],
        dtype=np.uint8,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one image as Original / GT / SAM3 / TTST+SAM3 / "
            "Real-ESRGAN+SAM3."
        )
    )
    parser.add_argument("--img", required=True, help="Input image path.")
    parser.add_argument("--gt", required=True, help="GT mask path.")
    parser.add_argument(
        "--dataset",
        choices=["dlrsd", "isaid_split"],
        default="dlrsd",
        help="Palette and GT decoding rule.",
    )
    parser.add_argument(
        "--classname-path",
        default=None,
        help="Class prompt file. Default follows --dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="resources/image_gt_sam3_ttst_vis",
        help="Output directory.",
    )
    parser.add_argument("--sr-ckpt-path", default="weights/TTST/ttst_4x.pth")
    parser.add_argument("--sr-input-size", type=int, default=256)
    parser.add_argument("--sr-dim", type=int, default=180)
    parser.add_argument(
        "--realesrgan-model-path",
        default="weights/Real-ESRGAN/RealESRGAN_x4plus.pth",
        help="Real-ESRGAN x4plus checkpoint path.",
    )
    parser.add_argument("--realesrgan-tile", type=int, default=0)
    parser.add_argument("--realesrgan-half", action="store_true", default=True)
    parser.add_argument("--prob-thd", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--bg-idx", type=int, default=None)
    parser.add_argument("--slide-stride", type=int, default=0)
    parser.add_argument("--slide-crop", type=int, default=0)
    parser.add_argument("--allow-bicubic-fallback", action="store_true")
    return parser.parse_args()


def default_classname_path(dataset):
    if dataset == "dlrsd":
        return "./configs/cls_dlrsd.txt"
    if dataset == "isaid_split":
        return "./configs/cls_iSAID_split.txt"
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_eval_params(dataset, args):
    if dataset == "dlrsd":
        prob_thd = 0.0 if args.prob_thd is None else args.prob_thd
        confidence_threshold = (
            0.5 if args.confidence_threshold is None else args.confidence_threshold
        )
        bg_idx = 255 if args.bg_idx is None else args.bg_idx
    elif dataset == "isaid_split":
        prob_thd = 0.0 if args.prob_thd is None else args.prob_thd
        confidence_threshold = (
            0.4 if args.confidence_threshold is None else args.confidence_threshold
        )
        bg_idx = 255 if args.bg_idx is None else args.bg_idx
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return prob_thd, confidence_threshold, bg_idx


def build_data(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to("cuda")
    data_sample = SegDataSample()
    data_sample.set_metainfo(
        {
            "img_path": str(img_path),
            "ori_shape": img.size[::-1],
        }
    )
    return img, img_tensor, data_sample


def rgb_to_label(rgb, palette):
    label = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(palette):
        label[np.all(rgb == color.reshape(1, 1, 3), axis=2)] = class_id
    return label


def read_gt(gt_path, dataset):
    gt = np.array(Image.open(gt_path))
    palette = PALETTES[dataset]

    if gt.ndim == 3 and gt.shape[2] >= 3:
        rgb = gt[:, :, :3]
        if not (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            return rgb_to_label(rgb, palette)
        gt = rgb[:, :, 0]

    gt = gt.astype(np.int64)
    if dataset == "dlrsd":
        # DLRSD raw masks often use 0 as ignore and 1..17 as valid classes.
        if gt.max(initial=0) > len(palette) - 1:
            label = gt.copy()
            label[gt == 0] = 255
            valid = gt != 0
            label[valid] = gt[valid] - 1
            return label
    return gt


def colorize_label(label, dataset):
    label = np.asarray(label)
    palette = PALETTES[dataset]
    color = np.full((*label.shape, 3), 220, dtype=np.uint8)
    for class_id, rgb in enumerate(palette):
        color[label == class_id] = rgb
    return Image.fromarray(color, mode="RGB")


def build_sam3_model(classname_path, prob_thd, confidence_threshold, bg_idx, args):
    return SegEarthOV3Segmentation(
        type="SegEarthOV3Segmentation",
        model_type="SAM3",
        classname_path=classname_path,
        prob_thd=prob_thd,
        confidence_threshold=confidence_threshold,
        bg_idx=bg_idx,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
    )


def build_ttst_sam3_model(classname_path, prob_thd, confidence_threshold, bg_idx, args):
    return SegEarthOV3SegmentationSRInput(
        type="SegEarthOV3SegmentationSRInput",
        model_type="SAM3_SRInput",
        classname_path=classname_path,
        prob_thd=prob_thd,
        confidence_threshold=confidence_threshold,
        bg_idx=bg_idx,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
        use_sr_input=True,
        sr_backend="ttst",
        sr_ckpt_path=args.sr_ckpt_path,
        sr_input_size=args.sr_input_size,
        sr_dim=args.sr_dim,
        allow_bicubic_fallback=args.allow_bicubic_fallback,
        keep_sr_output=False,
        use_original_logit_merge=True,
    )


def build_realesrgan_sam3_model(
    classname_path, prob_thd, confidence_threshold, bg_idx, args
):
    return SegEarthOV3SegmentationSRInput(
        type="SegEarthOV3SegmentationSRInput",
        model_type="SAM3_SRInput",
        classname_path=classname_path,
        prob_thd=prob_thd,
        confidence_threshold=confidence_threshold,
        bg_idx=bg_idx,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
        use_sr_input=True,
        sr_backend="realesrgan",
        realesrgan_model_name="RealESRGAN_x4plus",
        realesrgan_model_path=args.realesrgan_model_path,
        realesrgan_outscale=4,
        realesrgan_tile=args.realesrgan_tile,
        realesrgan_tile_pad=10,
        realesrgan_pre_pad=0,
        realesrgan_half=args.realesrgan_half,
        realesrgan_gpu_id=None,
        allow_bicubic_fallback=args.allow_bicubic_fallback,
        keep_sr_output=False,
        use_original_logit_merge=True,
    )


@torch.no_grad()
def predict(model, img_tensor, data_sample):
    pred = model.predict(img_tensor, data_samples=[data_sample])
    return pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)


def save_outputs(img, gt, sam3_pred, ttst_pred, realesrgan_pred, dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_color = colorize_label(gt, dataset)
    sam3_color = colorize_label(sam3_pred, dataset)
    ttst_color = colorize_label(ttst_pred, dataset)
    realesrgan_color = colorize_label(realesrgan_pred, dataset)

    img.save(out_dir / "01_original.png")
    gt_color.save(out_dir / "02_gt_seg.png")
    sam3_color.save(out_dir / "03_sam3_seg.png")
    ttst_color.save(out_dir / "04_ttst_sam3_seg.png")
    realesrgan_color.save(out_dir / "05_realesrgan_sam3_seg.png")

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    panels = [
        (img, "Original"),
        (gt_color, "GT"),
        (sam3_color, "SAM3"),
        (ttst_color, "TTST + SAM3"),
        (realesrgan_color, "Real-ESRGAN + SAM3"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    montage_path = out_dir / "montage_original_gt_sam3_ttst_realesrgan.png"
    plt.savefig(montage_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return montage_path


def main():
    args = parse_args()
    img_path = Path(args.img)
    gt_path = Path(args.gt)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    classname_path = args.classname_path or default_classname_path(args.dataset)
    prob_thd, confidence_threshold, bg_idx = default_eval_params(args.dataset, args)

    img, img_tensor, data_sample = build_data(img_path)
    gt = read_gt(gt_path, args.dataset)

    sam3_model = build_sam3_model(
        classname_path, prob_thd, confidence_threshold, bg_idx, args
    )
    ttst_model = build_ttst_sam3_model(
        classname_path, prob_thd, confidence_threshold, bg_idx, args
    )
    realesrgan_model = build_realesrgan_sam3_model(
        classname_path, prob_thd, confidence_threshold, bg_idx, args
    )

    try:
        sam3_pred = predict(sam3_model, img_tensor, data_sample)
        ttst_pred = predict(ttst_model, img_tensor, data_sample)
        realesrgan_pred = predict(realesrgan_model, img_tensor, data_sample)
    finally:
        del sam3_model
        del ttst_model
        del realesrgan_model
        torch.cuda.empty_cache()

    sample_out_dir = Path(args.out_dir) / img_path.stem
    montage_path = save_outputs(
        img, gt, sam3_pred, ttst_pred, realesrgan_pred, args.dataset, sample_out_dir
    )

    ttst_diff = sam3_pred != ttst_pred
    realesrgan_diff = sam3_pred != realesrgan_pred
    print(f"Image: {img_path}")
    print(f"GT   : {gt_path}")
    print(f"Saved montage to: {montage_path}")
    print(f"GT unique labels: {np.unique(gt)}")
    print(f"SAM3 unique labels: {np.unique(sam3_pred)}")
    print(f"TTST+SAM3 unique labels: {np.unique(ttst_pred)}")
    print(f"Real-ESRGAN+SAM3 unique labels: {np.unique(realesrgan_pred)}")
    print(f"SAM3 vs TTST+SAM3 diff ratio: {float(ttst_diff.mean()):.6f}")
    print(f"SAM3 vs TTST+SAM3 changed pixels: {int(ttst_diff.sum())}")
    print(
        "SAM3 vs Real-ESRGAN+SAM3 diff ratio: "
        f"{float(realesrgan_diff.mean()):.6f}"
    )
    print(
        "SAM3 vs Real-ESRGAN+SAM3 changed pixels: "
        f"{int(realesrgan_diff.sum())}"
    )


if __name__ == "__main__":
    main()
