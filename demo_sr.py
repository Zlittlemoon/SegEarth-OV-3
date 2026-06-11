import os
from pathlib import Path

# These env vars must be set before building SegEarthOV3SegmentationSoftPrompt.
# This demo intentionally uses the original SAM3 weights only, without a fine-tuned
# SR adapter checkpoint. The larger SR scales are only for visualizing whether the
# untrained SR coupling path can affect inference.
os.environ["SAM3_RS_USE_SR"] = "1"
os.environ.setdefault("SAM3_RS_SR_CKPT", "weights/TTST/ttst_4x.pth")
os.environ.setdefault("SAM3_RS_SR_INPUT_SIZE", "256")
os.environ.setdefault("SAM3_RS_SR_DIM", "180")
os.environ["SAM3_RS_SR_RES_SCALE"] = "0.5"
os.environ["SAM3_RS_SR_INIT_SCALE"] = "1.0"
os.environ["SAM3_RS_PROMPT_TUNING"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mmseg.structures import SegDataSample
from torchvision import transforms

from segearthov3_segmentor_merge import SegEarthOV3Segmentation
from segearthov3_segmentor_merge_softprompt import SegEarthOV3SegmentationSoftPrompt


DATA_ROOT = Path("data/isaid_patch")
IMG_PATH = Path("data/isaid_patch/img_dir/val/P1651_1024_1920_3072_3968.png")
GT_PATH = Path("data/isaid_patch/ann_dir/val/P1651_1024_1920_3072_3968_instance_color_RGB.png")
CLASSNAME_PATH = "configs/cls_iSAID.txt"
OUT_PATH = "resources/isaid_patch_data/isaid_patch/img_dir/val/P1651_1024_1920_3072_3968_sr.png"

NAME_LIST = [
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

ISAID_COLOR_TO_ID = {
    (0, 0, 63): 1,        # ship
    (0, 63, 63): 2,       # storage tank
    (0, 63, 0): 3,        # baseball diamond
    (0, 63, 127): 4,      # tennis court
    (0, 63, 191): 5,      # basketball court
    (0, 63, 255): 6,      # ground track field
    (0, 127, 63): 7,      # bridge
    (0, 127, 127): 8,     # large vehicle
    (0, 0, 127): 9,       # small vehicle
    (0, 0, 191): 10,      # helicopter
    (0, 0, 255): 11,      # swimming pool
    (0, 191, 127): 12,    # roundabout
    (0, 127, 191): 13,    # soccer ball field
    (0, 127, 255): 14,    # plane
    (0, 100, 155): 15,    # harbor
}


def write_class_names(path, names):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names))


def resolve_isaid_sample():
    if IMG_PATH is not None and GT_PATH is not None:
        return Path(IMG_PATH), Path(GT_PATH)

    img_dir = DATA_ROOT / "img_dir" / "val"
    gt_dir = DATA_ROOT / "ann_dir" / "val"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"iSAID image dir not found: {img_dir}. Set IMG_PATH/GT_PATH in demo_sr.py."
        )
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"iSAID annotation dir not found: {gt_dir}. Set IMG_PATH/GT_PATH in demo_sr.py."
        )

    image_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    for img_path in image_paths:
        gt_path = gt_dir / f"{img_path.stem}.png"
        if gt_path.exists():
            return img_path, gt_path
        color_gt_path = gt_dir / f"{img_path.stem}_instance_color_RGB.png"
        if color_gt_path.exists():
            return img_path, color_gt_path
        same_suffix_gt = gt_dir / img_path.name
        if same_suffix_gt.exists():
            return img_path, same_suffix_gt

    raise FileNotFoundError(
        f"No matching GT found for images in {img_dir}. Set IMG_PATH/GT_PATH explicitly."
    )


def build_data(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to("cuda")

    data_sample = SegDataSample()
    data_sample.set_metainfo(
        {
            "img_path": str(img_path),
            "ori_shape": img.size[::-1],  # H, W
        }
    )
    return img, img_tensor, data_sample


def read_gt(gt_path):
    gt = np.array(Image.open(gt_path))
    # iSAID config uses reduce_zero_label=False and cls_iSAID.txt includes background.
    # Single-channel GT should already be aligned as 0..15.
    if gt.ndim == 2:
        return gt.astype(np.int64)

    if gt.ndim == 3 and gt.shape[2] >= 3:
        rgb = gt[:, :, :3]

        # Some converted masks are stored as RGB but all channels are identical.
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
            rgb[:, :, 1], rgb[:, :, 2]
        ):
            return rgb[:, :, 0].astype(np.int64)

        label = np.zeros(rgb.shape[:2], dtype=np.int64)
        matched = np.zeros(rgb.shape[:2], dtype=bool)
        for color, class_id in ISAID_COLOR_TO_ID.items():
            color_arr = np.array(color, dtype=rgb.dtype).reshape(1, 1, 3)
            mask = np.all(rgb == color_arr, axis=2)
            label[mask] = class_id
            matched |= mask

        # Black is background. Other unmatched colors are likely unexpected instance colors;
        # mark them ignored so they do not look like background in the visualization.
        black = np.all(rgb == np.array([0, 0, 0], dtype=rgb.dtype).reshape(1, 1, 3), axis=2)
        unknown = (~matched) & (~black)
        if unknown.any():
            label[unknown] = 255
            print(
                f"[GT WARNING] {int(unknown.sum())} pixels have unknown iSAID RGB colors; "
                "marked as 255(ignore)."
            )
        return label

    raise ValueError(f"Unsupported GT shape: {gt.shape}")


@torch.no_grad()
def predict_sam3(img_tensor, data_sample):
    model = SegEarthOV3Segmentation(
        type="SegEarthOV3Segmentation",
        model_type="SAM3",
        classname_path=CLASSNAME_PATH,
        prob_thd=0.5,
        confidence_threshold=0.4,
        slide_stride=0,
        slide_crop=0,
    )
    pred = model.predict(img_tensor, data_samples=[data_sample])
    pred = pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
    del model
    torch.cuda.empty_cache()
    return pred


@torch.no_grad()
def predict_sam3_sr(img_tensor, data_sample):
    model = SegEarthOV3SegmentationSoftPrompt(
        type="SegEarthOV3SegmentationSoftPrompt",
        model_type="SAM3",
        classname_path=CLASSNAME_PATH,
        prob_thd=0.5,
        confidence_threshold=0.4,
        slide_stride=0,
        slide_crop=0,
        checkpoint_path="weights/sam3/sam3.pt",
        finetuned_checkpoint_path=None,
        use_soft_prompt=True,
    )
    pred = model.predict(img_tensor, data_samples=[data_sample])
    pred = pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
    del model
    torch.cuda.empty_cache()
    return pred


def show_label(ax, label_map, title, vmax):
    label_map = np.asarray(label_map)
    display = label_map.astype(float)
    display[label_map == 255] = np.nan
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="lightgray")
    ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def main():
    write_class_names(CLASSNAME_PATH, NAME_LIST)
    img_path, gt_path = resolve_isaid_sample()
    print(f"Image: {img_path}")
    print(f"GT   : {gt_path}")
    print(
        "SR coupling scale: "
        f"init={os.environ['SAM3_RS_SR_INIT_SCALE']}, "
        f"res={os.environ['SAM3_RS_SR_RES_SCALE']} "
        "(untrained adapter, visualization only)"
    )

    img, img_tensor, data_sample = build_data(img_path)
    gt = read_gt(gt_path)

    sam3_pred = predict_sam3(img_tensor, data_sample)
    sam3_sr_pred = predict_sam3_sr(img_tensor, data_sample)

    diff = sam3_pred != sam3_sr_pred
    print(f"SAM3 vs SAM3+SR diff ratio: {float(diff.mean()):.6f}")
    print(f"SAM3 vs SAM3+SR changed pixels: {int(diff.sum())}")

    non_ignore_gt = gt[gt != 255]
    gt_max = int(non_ignore_gt.max()) if non_ignore_gt.size > 0 else 0
    vmax = max(len(NAME_LIST) - 1, gt_max, int(sam3_pred.max()), int(sam3_sr_pred.max()))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    show_label(axes[1], gt, "GT", vmax)
    show_label(axes[2], sam3_pred, "SAM3", vmax)
    show_label(axes[3], sam3_sr_pred, "SAM3 + SR Coupling", vmax)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved visualization to: {OUT_PATH}")
    print(f"GT unique labels: {np.unique(gt)}")
    print(f"SAM3 unique labels: {np.unique(sam3_pred)}")
    print(f"SAM3+SR unique labels: {np.unique(sam3_sr_pred)}")


if __name__ == "__main__":
    main()
