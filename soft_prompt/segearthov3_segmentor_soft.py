import numpy as np
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

# 按你当前 builder 文件的定义来
# 如果你项目里导出路径是 from sam3 import build_rs_prompt_semantic_model
# 就把这一行替换掉
from sam3.model_builder import build_rs_prompt_semantic_model


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()

    class_names, class_indices = [], []
    for idx, line in enumerate(name_sets):
        names_i = [x.strip() for x in line.split(',')]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]

    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices


@MODELS.register_module()
class SegEarthOV3SoftPromptSegmentation(BaseSegmentor):
    """
    MMSeg wrapper for evaluating soft-prompt tuned RSPromptSemanticModel on LoveDA.

    核心变化：
    1. 不使用 Sam3Processor
    2. 直接调用 build_rs_prompt_semantic_model(...)
    3. 加载 trainer checkpoint['model']
    4. 推理输入构造成 RSPromptSemanticModel.forward() 真实要求的 batch 结构
    """

    def __init__(
        self,
        classname_path,
        sam3_base_ckpt='weights/sam3/sam3.pt',
        soft_prompt_ckpt='outputs/dlrsd_prompt_soft_only_1X4090/checkpoints/checkpoint_2.pt',
        bpe_path='./sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        device='cuda',
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        **kwargs
    ):
        super().__init__()

        self.device = torch.device(device)
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.image_size = 1008
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.tensor(self.query_idx, dtype=torch.int64, device=self.device)

        # 1) 先构建基础 SAM3 + RSPromptSemanticModel wrapper
        # model_builder.py 已支持 **kwargs 继续传给 build_sam3_image_model
        # 所以这里显式传 checkpoint_path 很重要
        self.model = build_rs_prompt_semantic_model(
            bpe_path=bpe_path,
            checkpoint_path=sam3_base_ckpt,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )

        # 2) 再加载 soft-prompt 训练后的 trainer checkpoint
        ckpt = torch.load(soft_prompt_ckpt, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print('[SoftPromptEval] loaded soft prompt ckpt:', soft_prompt_ckpt)
        print('[SoftPromptEval] missing keys:', len(missing))
        if len(missing) > 0:
            print('[SoftPromptEval] first missing keys:', missing[:20])
        print('[SoftPromptEval] unexpected keys:', len(unexpected))
        if len(unexpected) > 0:
            print('[SoftPromptEval] first unexpected keys:', unexpected[:20])

        self.model.to(self.device)
        self.model.eval()

    def _model_device(self):
        return next(self.model.parameters()).device
    
    def _preprocess_pil(self, image: Image.Image):
        """
        Match SAM3 builder settings:
        - image_size = 1008
        - image_mean = (0.5, 0.5, 0.5)
        - image_std  = (0.5, 0.5, 0.5)

        Return:
        tensor: [1, 3, 1008, 1008]
        orig_hw: original PIL size as (h, w)
        """
        device = self._model_device()

        image = image.convert("RGB")
        orig_w, orig_h = image.size

        # 关键修复：固定 resize 到 SAM3 期望输入尺寸
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        x = np.array(image).astype(np.float32) / 255.0  # [H,W,3], [0,1]
        x = torch.from_numpy(x).permute(2, 0, 1)        # [3,H,W]

        # builder 里写的是 mean=std=0.5
        x = (x - 0.5) / 0.5

        x = x.unsqueeze(0).to(device)                   # [1,3,1008,1008]
        return x, (orig_h, orig_w)


    def _build_minimal_find_input(self, device):
        return SimpleNamespace(
            text_ids=torch.tensor([0], dtype=torch.long, device=device),
            img_ids=torch.tensor([0], dtype=torch.long, device=device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
        )


    def _make_minimal_batch(self, image_tensor, query_text):
        """
        image_tensor: [1,3,1008,1008]
        """
        device = self._model_device()
        batch = SimpleNamespace(
            img_batch=image_tensor,
            find_text_batch=[query_text],
            find_input=self._build_minimal_find_input(device),
        )
        return batch


    def _extract_query_logits(self, model_out, out_h, out_w):
        """
        model_out['semantic_seg'] is expected from RSPromptSemanticModel.forward()
        and is typically [B,1,H,W].
        """
        pred = model_out["semantic_seg"]

        if pred.dim() == 4:
            pred = pred[0, 0]
        elif pred.dim() == 3:
            pred = pred[0]
        elif pred.dim() != 2:
            raise RuntimeError(
                f"Unexpected semantic_seg shape: {tuple(model_out['semantic_seg'].shape)}"
            )

        # 关键：把 1008x1008 输出 resize 回当前 view/crop 的原始尺寸
        if tuple(pred.shape) != (out_h, out_w):
            pred = F.interpolate(
                pred[None, None],
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        return pred


    def _inference_single_view(self, image):
        """
        image: PIL image, can be full image or a sliding-window crop
        returns:
        seg_logits: [num_queries, h, w] in the original view size
        """
        w, h = image.size
        device = self._model_device()
        seg_logits = torch.zeros((self.num_queries, h, w), device=device)

        image_tensor, _ = self._preprocess_pil(image)

        autocast_enabled = device.type == "cuda"

        with torch.no_grad():
            if autocast_enabled:
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx:
                for query_i, query_word in enumerate(self.query_words):
                    batch = self._make_minimal_batch(image_tensor, query_word)
                    out = self.model(batch)
                    query_logit = self._extract_query_logits(out, h, w)
                    seg_logits[query_i] = query_logit

        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """
        滑窗推理，逻辑延续原始 segmentor 的写法。
        """
        w_img, h_img = image.size
        device = self._model_device()

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros((self.num_queries, h_img, w_img), device=device)
        count_mat = torch.zeros((1, h_img, w_img), device=device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)

                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        preds = preds / count_mat.clamp(min=1.0)
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]
                )
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            image_path = meta['img_path']
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            if (
                self.slide_crop is not None
                and self.slide_crop > 0
                and (
                    self.slide_crop < image.size[0]
                    or self.slide_crop < image.size[1]
                )
            ):
                seg_logits = self.slide_inference(
                    image=image,
                    stride=self.slide_stride,
                    crop_size=self.slide_crop,
                )
            else:
                seg_logits = self._inference_single_view(image)

            if tuple(seg_logits.shape[-2:]) != tuple(ori_shape):
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            # 多 synonym query -> 同一 semantic class 聚合
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx, num_classes=self.num_cls)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = torch.argmax(seg_logits, dim=0)
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(data=seg_logits),
                'pred_sem_seg': PixelData(data=seg_pred.unsqueeze(0)),
            })

        return data_samples

    def _forward(self, inputs, data_samples=None):
        return self.predict(inputs, data_samples)

    def inference(self, img, batch_img_metas):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def loss(self, inputs, data_samples):
        raise NotImplementedError