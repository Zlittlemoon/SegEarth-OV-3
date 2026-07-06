"""Visualize overlap between a CLIP/RemoteCLIP dense text-similarity heatmap
and a SAM3 coarse mask for a single (image, text prompt) pair.

Usage:
    python reprompt/visualize_clip_sam_overlap.py \
        --image data/loveda/Images/1366.png \
        --text water \
        --out show_dir/clip_sam_overlap.png
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import matplotlib.pyplot as plt

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to a single RGB image')
    parser.add_argument('--text', required=True, help='class text prompt, e.g. "water"')

    parser.add_argument('--clip-model', default='ViT-B-32')
    parser.add_argument('--clip-ckpt', default='weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt')
    parser.add_argument('--clip-input-size', type=int, default=224)
    parser.add_argument('--clip-thd-percentile', type=float, default=70.0,
                         help='percentile of the CLIP similarity map used as the binarization threshold')

    parser.add_argument('--sam-ckpt', default='weights/sam3/sam3.pt')
    parser.add_argument('--sam-bpe', default='./sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    parser.add_argument('--sam-thd', type=float, default=0.5,
                         help='absolute sigmoid probability threshold used to binarize the SAM3 coarse '
                              'logit. Ignored if --sam-thd-percentile is set. A single-class raw sigmoid '
                              'is not calibrated the way the real pipeline\'s cross-class argmax is, so '
                              'this absolute threshold can select most of the image; prefer '
                              '--sam-thd-percentile for a sanity check.')
    parser.add_argument('--sam-thd-percentile', type=float, default=None,
                         help='if set, binarize the SAM3 mask using this percentile of its own '
                              'probability map instead of the absolute --sam-thd, so it is directly '
                              'comparable to the CLIP percentile mask')

    parser.add_argument('--out', default='show_dir/clip_sam_overlap.png')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLIP / RemoteCLIP dense heatmap
# ---------------------------------------------------------------------------

def load_clip_model(model_name, ckpt_path, device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    message = model.load_state_dict(ckpt)
    print(f'[CLIP] loaded {ckpt_path}: {message}')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def clip_dense_similarity(model, preprocess, tokenizer, image, text, device):
    """MaskCLIP-style dense similarity map.

    Naively reusing the last transformer block's own output (post QK-softmax
    attention) gives very poorly localized maps, because that block's self
    attention was only ever trained to mix information *into* the pooled CLS
    token, not to keep each patch token spatially discriminative. The MaskCLIP
    fix (Zhou et al., ECCV'22) is to skip the query-key softmax attention in
    just the last block and keep only the value projection, so each patch's
    output only depends on itself.
    """
    visual = model.visual
    patch_size = visual.conv1.kernel_size[0]
    input_res = preprocess.transforms[0].size
    if isinstance(input_res, (tuple, list)):
        input_res = input_res[0]
    grid = input_res // patch_size

    last_block = visual.transformer.resblocks[-1]
    captured = {}

    def _hook(_module, inputs):
        captured['tokens_in'] = inputs[0]

    handle = last_block.register_forward_pre_hook(_hook)

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        visual(img_tensor)
        handle.remove()

        tokens = captured['tokens_in']
        batch_size = img_tensor.shape[0]
        if tokens.shape[0] == batch_size:
            pass  # already [batch, seq_len, width] (NLD)
        elif tokens.shape[1] == batch_size:
            tokens = tokens.permute(1, 0, 2)  # [seq_len, batch, width] (LND) -> NLD
        else:
            raise RuntimeError(f'unrecognized resblock input shape {tuple(tokens.shape)} '
                                f'for batch_size={batch_size}')

        # value-only "attention": skip QK softmax mixing across patches
        ln1_out = last_block.ln_1(tokens)
        attn = last_block.attn
        d = ln1_out.shape[-1]
        v_weight = attn.in_proj_weight[2 * d:3 * d, :]
        v_bias = attn.in_proj_bias[2 * d:3 * d]
        value = F.linear(ln1_out, v_weight, v_bias)
        attn_out = attn.out_proj(value)
        tokens = tokens + attn_out
        tokens = tokens + last_block.mlp(last_block.ln_2(tokens))

        tokens = visual.ln_post(tokens)
        if visual.proj is not None:
            tokens = tokens @ visual.proj

        patch_tokens = tokens[:, 1:, :]  # drop CLS token
        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sim = (patch_tokens @ text_feat.T).squeeze(-1).squeeze(0)  # [seq_len]
        sim_map = sim.view(grid, grid).float().cpu().numpy()

    return sim_map


def resize_heatmap(sim_map, out_hw):
    h, w = out_hw
    t = torch.from_numpy(sim_map).view(1, 1, *sim_map.shape)
    t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


# ---------------------------------------------------------------------------
# SAM3 coarse mask (mirrors _inference_single_view's instance/semantic fusion)
# ---------------------------------------------------------------------------

def build_sam3_processor(bpe_path, ckpt_path, device):
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=ckpt_path, device=device)
    return Sam3Processor(model, confidence_threshold=0.5, device=device)


def sam3_coarse_logit(processor, image, text, device):
    w, h = image.size
    query_logit = torch.zeros((h, w), device=device)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=text)

        if state.get('masks_logits') is not None and state['masks_logits'].shape[0] > 0:
            for inst_id in range(state['masks_logits'].shape[0]):
                inst_logit = state['masks_logits'][inst_id].squeeze()
                inst_score = state['object_score'][inst_id]
                if inst_logit.shape != (h, w):
                    inst_logit = F.interpolate(
                        inst_logit.view(1, 1, *inst_logit.shape),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze()
                query_logit = torch.maximum(query_logit, inst_logit * inst_score)

        if 'semantic_mask_logits' in state:
            sem_logit = state['semantic_mask_logits'].squeeze()
            if sem_logit.shape != (h, w):
                sem_logit = F.interpolate(
                    sem_logit.view(1, 1, *sem_logit.shape),
                    size=(h, w), mode='bilinear', align_corners=False
                ).squeeze()
            query_logit = torch.maximum(query_logit, sem_logit)

    return query_logit


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def overlay(image_np, mask, color, alpha=0.45):
    out = image_np.copy()
    colored = np.zeros_like(out)
    colored[mask] = color
    out[mask] = (out[mask] * (1 - alpha) + colored[mask] * alpha).astype(np.uint8)
    return out


def extract_points_from_mask(mask, max_points=5):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, num_labels)]
    comps.sort(key=lambda t: t[0], reverse=True)
    return [c[1] for c in comps[:max_points]]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device = args.device

    image = Image.open(args.image).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    print('[CLIP] loading model...')
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(args.clip_model, args.clip_ckpt, device)
    sim_map = clip_dense_similarity(clip_model, clip_preprocess, clip_tokenizer, image, args.text, device)
    sim_map_full = resize_heatmap(sim_map, (h, w))
    clip_thd = np.percentile(sim_map_full, args.clip_thd_percentile)
    clip_mask = sim_map_full > clip_thd

    print('[SAM3] loading model...')
    processor = build_sam3_processor(args.sam_bpe, args.sam_ckpt, device)
    coarse_logit = sam3_coarse_logit(processor, image, args.text, device)
    sam_prob = torch.sigmoid(coarse_logit).cpu().numpy()

    percentiles = [1, 5, 25, 50, 75, 90, 95, 99]
    pvals = np.percentile(sam_prob, percentiles)
    print('[SAM3 prob distribution] ' +
          '  '.join(f'p{p}={v:.3f}' for p, v in zip(percentiles, pvals)))

    if args.sam_thd_percentile is not None:
        sam_thd = np.percentile(sam_prob, args.sam_thd_percentile)
        sam_mask = sam_prob > sam_thd
        sam_title = f'SAM3 mask (top {100 - args.sam_thd_percentile:.0f}%, thd={sam_thd:.3f})'
    else:
        sam_mask = sam_prob > args.sam_thd
        sam_title = f'SAM3 mask (abs thd={args.sam_thd})'

    inter_mask = sam_mask & clip_mask
    union_mask = sam_mask | clip_mask
    inter_area, union_area = inter_mask.sum(), union_mask.sum()
    iou = inter_area / max(union_area, 1)
    points = extract_points_from_mask(inter_mask)

    print(f'[stats] SAM area={sam_mask.sum()}  CLIP area={clip_mask.sum()}  '
          f'intersection={inter_area}  union={union_area}  IoU={iou:.4f}  '
          f'positive points extracted={len(points)}')

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('original image')

    axes[0, 1].imshow(overlay(image_np, sam_mask, (255, 0, 0)))
    axes[0, 1].set_title(sam_title)

    im = axes[0, 2].imshow(sim_map_full, cmap='jet')
    axes[0, 2].set_title(f'CLIP dense heatmap: "{args.text}"')
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(overlay(image_np, clip_mask, (0, 0, 255)))
    axes[1, 0].set_title(f'CLIP mask (top {100 - args.clip_thd_percentile:.0f}%)')

    inter_vis = overlay(image_np, union_mask, (255, 255, 0))
    inter_vis = overlay(inter_vis, inter_mask, (0, 255, 0))
    for cx, cy in points:
        axes[1, 1].scatter([cx], [cy], marker='*', s=150, c='white', edgecolors='black')
    axes[1, 1].imshow(inter_vis)
    axes[1, 1].set_title('green=intersection  yellow=union-only  *=candidate positive points')

    axes[1, 2].axis('off')
    axes[1, 2].text(0.0, 0.5,
                     f'SAM area: {sam_mask.sum()}\n'
                     f'CLIP area: {clip_mask.sum()}\n'
                     f'Intersection: {inter_area}\n'
                     f'Union: {union_area}\n'
                     f'IoU: {iou:.4f}\n'
                     f'points: {len(points)}',
                     fontsize=13, va='center')

    for ax in axes.flat:
        ax.axis('off') if ax is not axes[1, 2] else None

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()