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


DLRSD_NAMES = [
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
]

DLRSD_PALETTE = np.array(
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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Original / GT / SAM3 / SR-input+SAM3 on DLRSD images."
    )
    parser.add_argument(
        "--img",
        default=None,
        help="Input image path. Default: first image under data/DLRSD_split/val/imgs.",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="GT mask path. Default: data/DLRSD_split/val/masks/<image_stem>.png.",
    )
    parser.add_argument(
        "--data-root",
        default="data/DLRSD_split",
        help="DLRSD split root used when --img/--gt are not specified.",
    )
    parser.add_argument(
        "--classname-path",
        default="./configs/cls_dlrsd.txt",
        help="Class name file used by SAM3 text prompts.",
    )
    parser.add_argument(
        "--out-dir",
        default="resources/sr_input_vis_dlrsd2",
        help="Directory for four separate outputs and the montage.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of DLRSD val images to visualize when --img is not specified.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in sorted DLRSD val image list when --img is not specified.",
    )
    parser.add_argument(
        "--sr-ckpt-path",
        default="weights/TTST/ttst_4x.pth",
        help="4x TTST checkpoint path.",
    )
    parser.add_argument("--sr-input-size", type=int, default=256)
    parser.add_argument("--sr-dim", type=int, default=180)
    parser.add_argument(
        "--allow-bicubic-fallback",
        action="store_true",
        help="Use bicubic 4x resize if TTST code/checkpoint is unavailable.",
    )
    parser.add_argument("--prob-thd", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--slide-stride", type=int, default=512)
    parser.add_argument("--slide-crop", type=int, default=512)
    return parser.parse_args()


def _resolve_gt_path(data_root, img_path, gt_path=None):
    if gt_path is not None:
        gt_path = Path(gt_path)
    else:
        gt_path = Path(data_root) / "val" / "masks" / f"{img_path.stem}.png"

    if not gt_path.exists():
        raise FileNotFoundError(f"GT mask not found: {gt_path}")
    return gt_path


def resolve_samples(data_root, img_path=None, gt_path=None, num_samples=1, start_index=0):
    if img_path is not None:
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return [(img_path, _resolve_gt_path(data_root, img_path, gt_path))]

    img_dir = Path(data_root) / "val" / "imgs"
    image_paths = sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {img_dir}")

    start_index = max(int(start_index), 0)
    end_index = min(start_index + max(int(num_samples), 1), len(image_paths))
    selected = image_paths[start_index:end_index]
    if not selected:
        raise IndexError(
            f"No image selected: start_index={start_index}, dataset_size={len(image_paths)}"
        )

    return [(path, _resolve_gt_path(data_root, path)) for path in selected]


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


def rgb_mask_to_label(rgb):
    label = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(DLRSD_PALETTE):
        label[np.all(rgb == color.reshape(1, 1, 3), axis=2)] = class_id
    return label


def read_dlrsd_gt(gt_path):
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3 and gt.shape[2] >= 3:
        rgb = gt[:, :, :3]
        if not (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            return rgb_mask_to_label(rgb)
        gt = rgb[:, :, 0]

    gt = gt.astype(np.int64)

    # Match DLRSDDataset(reduce_zero_label=True): raw 0 -> ignore, raw 1..17 -> 0..16.
    label = gt.copy()
    label[gt == 0] = 255
    valid = gt != 0
    label[valid] = gt[valid] - 1
    return label


def colorize_label(label):
    label = np.asarray(label)
    color = np.full((*label.shape, 3), 220, dtype=np.uint8)
    for class_id, rgb in enumerate(DLRSD_PALETTE):
        color[label == class_id] = rgb
    return Image.fromarray(color, mode="RGB")


def build_sam3_model(args):
    return SegEarthOV3Segmentation(
        type="SegEarthOV3Segmentation",
        model_type="SAM3",
        classname_path=args.classname_path,
        prob_thd=args.prob_thd,
        confidence_threshold=args.confidence_threshold,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
    )


def build_sr_sam3_model(args):
    return SegEarthOV3SegmentationSRInput(
        type="SegEarthOV3SegmentationSRInput",
        model_type="SAM3_SRInput",
        classname_path=args.classname_path,
        prob_thd=args.prob_thd,
        confidence_threshold=args.confidence_threshold,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
        use_sr_input=True,
        sr_ckpt_path=args.sr_ckpt_path,
        sr_input_size=args.sr_input_size,
        sr_dim=args.sr_dim,
        allow_bicubic_fallback=args.allow_bicubic_fallback,
        keep_sr_output=False,
        use_original_logit_merge=True,
    )


@torch.no_grad()
def predict_with_model(model, img_tensor, data_sample):
    pred = model.predict(img_tensor, data_samples=[data_sample])
    pred = pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
    return pred


def save_outputs(img, gt, sam3_pred, sr_sam3_pred, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_color = colorize_label(gt)
    sam3_color = colorize_label(sam3_pred)
    sr_sam3_color = colorize_label(sr_sam3_pred)

    img.save(out_dir / "01_original.png")
    gt_color.save(out_dir / "02_gt_seg.png")
    sam3_color.save(out_dir / "03_sam3_seg.png")
    sr_sam3_color.save(out_dir / "04_sr_sam3_seg.png")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    panels = [
        (img, "Original"),
        (gt_color, "GT"),
        (sam3_color, "SAM3"),
        (sr_sam3_color, "SR + SAM3"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    montage_path = out_dir / "montage_original_gt_sam3_sr_sam3.png"
    plt.savefig(montage_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return montage_path


def main():
    args = parse_args()
    samples = resolve_samples(
        args.data_root,
        args.img,
        args.gt,
        args.num_samples,
        args.start_index,
    )
    print(f"Selected {len(samples)} DLRSD sample(s).")

    sam3_model = build_sam3_model(args)
    sr_sam3_model = build_sr_sam3_model(args)

    try:
        for sample_idx, (img_path, gt_path) in enumerate(samples, start=1):
            print(f"[{sample_idx}/{len(samples)}] Image: {img_path}")
            print(f"[{sample_idx}/{len(samples)}] GT   : {gt_path}")

            img, img_tensor, data_sample = build_data(img_path)
            gt = read_dlrsd_gt(gt_path)

            sam3_pred = predict_with_model(sam3_model, img_tensor, data_sample)
            sr_sam3_pred = predict_with_model(sr_sam3_model, img_tensor, data_sample)

            diff = sam3_pred != sr_sam3_pred
            print(f"  SAM3 vs SR+SAM3 diff ratio: {float(diff.mean()):.6f}")
            print(f"  SAM3 vs SR+SAM3 changed pixels: {int(diff.sum())}")
            print(f"  GT unique labels: {np.unique(gt)}")
            print(f"  SAM3 unique labels: {np.unique(sam3_pred)}")
            print(f"  SR+SAM3 unique labels: {np.unique(sr_sam3_pred)}")

            sample_out_dir = Path(args.out_dir) / img_path.stem
            montage_path = save_outputs(
                img,
                gt,
                sam3_pred,
                sr_sam3_pred,
                sample_out_dir,
            )
            print(f"  Saved montage to: {montage_path}")
    finally:
        del sam3_model
        del sr_sam3_model
        torch.cuda.empty_cache()

    print(f"Saved all outputs under: {args.out_dir}")


if __name__ == "__main__":
    main()
