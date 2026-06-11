import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mmseg.structures import SegDataSample
from torchvision import transforms

from segearthov3_segmentor_original import SegEarthOV3Segmentation


ISAID_NAMES = [
    "background",
    "ship",
    "store tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "swimming pool",
    "roundabout",
    "soccer ball field",
    "plane",
    "harbor",
]

ISAID_PALETTE = np.array(
    [
        [0, 0, 0],
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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SAM3 semantic segmentation on iSAID full images."
    )
    parser.add_argument(
        "--data-root",
        default="data/isaid_full_image",
        help="iSAID full image root.",
    )
    parser.add_argument("--img", default=None, help="Input image path.")
    parser.add_argument("--gt", default=None, help="Optional GT mask path.")
    parser.add_argument(
        "--classname-path",
        default="./configs/cls_iSAID.txt",
        help="Class prompt file for 16-class iSAID full image.",
    )
    parser.add_argument(
        "--out-dir",
        default="resources/isaid_full_image_sam3_vis",
        help="Output directory.",
    )
    parser.add_argument("--prob-thd", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--slide-stride", type=int, default=0)
    parser.add_argument("--slide-crop", type=int, default=0)
    return parser.parse_args()


def resolve_sample(data_root, img_path=None, gt_path=None):
    data_root = Path(data_root)
    if img_path is None:
        img_dir = data_root / "img_dir" / "val"
        image_paths = sorted(
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under: {img_dir}")
        img_path = image_paths[0]
    else:
        img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if gt_path is None:
        gt_dir = data_root / "ann_dir" / "val"
        candidates = [
            gt_dir / f"{img_path.stem}_instance_color_RGB.png",
            gt_dir / f"{img_path.stem}.png",
            gt_dir / img_path.name,
        ]
        gt_path = next((p for p in candidates if p.exists()), None)
    else:
        gt_path = Path(gt_path)

    return img_path, gt_path


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


def rgb_to_label(rgb):
    label = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(ISAID_PALETTE):
        label[np.all(rgb == color.reshape(1, 1, 3), axis=2)] = class_id
    return label


def read_gt(gt_path):
    if gt_path is None or not Path(gt_path).exists():
        return None

    gt = np.array(Image.open(gt_path))
    if gt.ndim == 2:
        return gt.astype(np.int64)

    rgb = gt[:, :, :3]
    if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
        rgb[:, :, 1], rgb[:, :, 2]
    ):
        return rgb[:, :, 0].astype(np.int64)

    return rgb_to_label(rgb)


def colorize_label(label):
    label = np.asarray(label)
    color = np.full((*label.shape, 3), 220, dtype=np.uint8)
    for class_id, rgb in enumerate(ISAID_PALETTE):
        color[label == class_id] = rgb
    return Image.fromarray(color, mode="RGB")


def build_model(args):
    return SegEarthOV3Segmentation(
        type="SegEarthOV3Segmentation",
        model_type="SAM3",
        classname_path=args.classname_path,
        prob_thd=args.prob_thd,
        confidence_threshold=args.confidence_threshold,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
    )


@torch.no_grad()
def predict(model, img_tensor, data_sample):
    pred = model.predict(img_tensor, data_samples=[data_sample])
    return pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)


def save_outputs(img, gt, pred, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_color = colorize_label(pred)
    img.save(out_dir / "01_original.png")
    pred_color.save(out_dir / "03_sam3_seg.png")

    if gt is not None:
        gt_color = colorize_label(gt)
        gt_color.save(out_dir / "02_gt_seg.png")
        panels = [(img, "Original"), (gt_color, "GT"), (pred_color, "SAM3")]
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        montage_path = out_dir / "montage_original_gt_sam3.png"
    else:
        panels = [(img, "Original"), (pred_color, "SAM3")]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        montage_path = out_dir / "montage_original_sam3.png"

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(montage_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    return montage_path


def main():
    args = parse_args()
    img_path, gt_path = resolve_sample(args.data_root, args.img, args.gt)
    print(f"Image: {img_path}")
    print(f"GT   : {gt_path if gt_path is not None else 'not found'}")

    img, img_tensor, data_sample = build_data(img_path)
    gt = read_gt(gt_path)

    model = build_model(args)
    try:
        pred = predict(model, img_tensor, data_sample)
    finally:
        del model
        torch.cuda.empty_cache()

    sample_out_dir = Path(args.out_dir) / img_path.stem
    montage_path = save_outputs(img, gt, pred, sample_out_dir)
    print(f"Saved montage to: {montage_path}")
    if gt is not None:
        print(f"GT unique labels: {np.unique(gt)}")
    print(f"SAM3 unique labels: {np.unique(pred)}")


if __name__ == "__main__":
    main()
