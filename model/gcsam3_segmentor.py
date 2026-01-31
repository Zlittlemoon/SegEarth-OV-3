import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from mmseg.models.losses import DiceLoss
from mmseg.registry import MODELS
from PIL import Image
from typing import Optional, List, Dict, Tuple, Any

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from Offline_database.prototype_utils import PrototypeManager
from adapter.adapter import RemoteSensingAdapter

@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 clip_ckpt_path="weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
                 database_path="Offline_database/Million-AID_train_image_features.pt",
                 adapter_topk=5,
                 **kwargs):
        super().__init__()
        
        self.device = device
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        
        self.query_words, self.query_class_ids = get_cls_idx(classname_path)
        self.query_idx = torch.tensor(self.query_class_ids, dtype=torch.int64, device=device)
        self.num_cls = int(self.query_idx.max().item()) + 1
        self.num_queries = len(self.query_words)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        
        self.proto_manager = PrototypeManager(
            clip_ckpt_path=clip_ckpt_path,
            database_path=database_path,
            device=device
        )
        
        self.adapter = RemoteSensingAdapter(topk=adapter_topk).to(device)

        # 缓存原型（squeeze 掉 batch 维度，adapter 会自动处理广播）
        self.cached_prototypes = {}
        for word in self.query_words:
            v_ref, scores = self.proto_manager.get_prototypes([word], topk=adapter_topk)
            # Squeeze batch dimension: [1, topk, 512] -> [topk, 512], [1, topk] -> [topk]
            self.cached_prototypes[word] = (v_ref.squeeze(0), scores.squeeze(0))

    @staticmethod
    def _get_fpn(backbone_out):
        if "backbone_fpn" in backbone_out:
            return backbone_out["backbone_fpn"]
        raise RuntimeError("Cannot find backbone_fpn in backbone_out.")

    @staticmethod
    def _set_fpn(backbone_out, fpn_list):
        backbone_out["backbone_fpn"] = fpn_list

    def _get_state_value(self, state, keys):
        """容错获取 state 中的值"""
        for k in keys:
            if k in state and state[k] is not None:
                return state[k]
        return None

    def _inference_single_view(self, image):
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            backbone_out = inference_state["backbone_out"]
            fpn_feats_orig = self._get_fpn(backbone_out)
            fpn_backup = [f.clone() for f in fpn_feats_orig]  # 深拷贝备份

            for q_idx, query_word in enumerate(self.query_words):
                # 每次循环重新克隆备份
                fpn_feats = [f.clone() for f in fpn_backup]
                
                # 获取原型（缓存的是 [1, topk, 512]）
                prototypes, proto_scores = self.cached_prototypes[query_word]
                
                # 应用 adapter
                enhanced_fpn = self.adapter(fpn_feats, prototypes, proto_scores)
                self._set_fpn(backbone_out, enhanced_fpn)
                
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                # 容错获取输出
                masks_logits = self._get_state_value(inference_state, 
                    ("masks_logits", "pred_masks", "pred_masks_logits", "masks"))
                object_score = self._get_state_value(inference_state, 
                    ("object_score", "pred_iou_scores", "iou_scores"))
                semantic_logits = self._get_state_value(inference_state, 
                    ("semantic_mask_logits", "semantic_logits", "pred_semantic_logits"))
                presence_score = self._get_state_value(inference_state, 
                    ("presence_score", "pred_presence_score"))

                if self.use_transformer_decoder and masks_logits is not None:
                    # 兼容多种形状
                    if masks_logits.ndim == 4:  # [N,1,H,W] 或 [B,N,H,W]
                        if masks_logits.shape[0] == 1:
                            masks_logits = masks_logits.squeeze(0)
                    if masks_logits.ndim == 3 and masks_logits.shape[0] > 1:  # [N,H,W]
                        for inst_id in range(masks_logits.shape[0]):
                            inst_logit = masks_logits[inst_id]
                            score = object_score[inst_id] if object_score is not None else 1.0
                            if inst_logit.shape != (h, w):
                                inst_logit = F.interpolate(inst_logit[None, None], size=(h, w), mode='bilinear', align_corners=False).squeeze()
                            seg_logits[q_idx] = torch.max(seg_logits[q_idx], inst_logit * score)

                if self.use_sem_seg and semantic_logits is not None:
                    sem = semantic_logits.squeeze()  # 去掉多余维度 → [H, W] 或 [1, H, W] → squeeze
                    if sem.dim() == 3 and sem.shape[0] == 1:
                        sem = sem.squeeze(0)
                    seg_logits[q_idx] = torch.max(seg_logits[q_idx], sem)

                if self.use_presence_score and presence_score is not None:
                    seg_logits[q_idx] *= presence_score

        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
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

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
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
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
                image_path = meta.get('img_path')
                image = Image.open(image_path).convert('RGB')
                ori_shape = meta['ori_shape']

                # [FIX 1] 使用与 baseline 相同的条件推理逻辑
                # 只有当 crop_size 小于图像尺寸时才使用滑动窗口
                if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                    seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
                else:
                    seg_logits = self._inference_single_view(image)

                # 如果累积后尺寸不对，resize
                if seg_logits.shape[-2:] != ori_shape:
                    seg_logits = F.interpolate(
                        seg_logits.unsqueeze(0),
                        size=ori_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                # [FIX 2] 使用与 baseline 完全相同的一词多类处理逻辑
                if self.num_cls != self.num_queries:
                    seg_logits = seg_logits.unsqueeze(0)  # [1, Q, H, W]
                    cls_index = F.one_hot(self.query_idx)  # [Q, C]
                    cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)  # [C, Q, 1, 1]
                    seg_logits = (seg_logits * cls_index).max(1)[0]  # [C, H, W]
                    seg_pred = seg_logits.argmax(0, keepdim=True)  # [1, H, W]

                # [FIX] Baseline 在 if 块外重新计算 seg_pred（虽然看起来是 bug，但要保持一致）
                seg_pred = torch.argmax(seg_logits, dim=0)  # [H, W]

                # [FIX 3] 应用概率阈值（与 baseline 一致）
                max_vals = seg_logits.max(0)[0]  # [H, W]
                seg_pred[max_vals < self.prob_thd] = self.bg_idx

                # 设置 pred_sem_seg（需要 unsqueeze 到 [1, H, W]）
                data_samples[i].set_data({
                    'seg_logits': PixelData(**{'data': seg_logits}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
                })

        return data_samples
            
    def _compose_query_logit_baseline(
        self,
        masks_logits: Optional[torch.Tensor],
        object_score: Optional[torch.Tensor],
        semantic_logits: Optional[torch.Tensor],
        presence_score: Optional[torch.Tensor],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        H, W = out_hw
        B = masks_logits.shape[0] if masks_logits is not None else 1

        seg = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)

        # instance masks（向量化上采样 + max）
        if self.use_transformer_decoder and masks_logits is not None:
            m = masks_logits  # [B, N, H_low, W_low]
            if m.ndim == 5:
                m = m.squeeze(2)
            if m.ndim == 4 and m.shape[1] == 1:
                m = m.squeeze(1)

            scores = object_score if object_score is not None else torch.ones((B, m.shape[1]), device=self.device)

            # 一次性上采样所有 instance
            m_up = F.interpolate(
                m.view(B * m.shape[1], 1, m.shape[2], m.shape[3]), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).view(B, m.shape[1], H, W)  # [B, N, H, W]

            # 加权
            m_up = m_up * scores.view(B, m.shape[1], 1, 1)

            # 对 N 维度取 max
            seg = torch.max(m_up, dim=1)[0]  # [B, H, W]

        # semantic（单次上采样）
        if self.use_sem_seg and semantic_logits is not None:
            sem = semantic_logits
            if sem.ndim == 4 and sem.shape[1] == 1:
                sem = sem.squeeze(1)

            sem_up = F.interpolate(
                sem.view(B, 1, sem.shape[1], sem.shape[2]), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # [B, H, W]

            seg = torch.max(seg, sem_up)

        # presence
        if self.use_presence_score and presence_score is not None:
            # presence_score may be [B] or [B,1] or [B, Q]. Ensure [B, 1, 1] when per-query already folded.
            if presence_score.dim() == 1:
                ps = presence_score.view(B, 1, 1)
            else:
                ps = presence_score.view(B, 1, 1)
            seg = seg * ps

        return seg
    
    def _forward(self, inputs, data_samples):
        pass  # Stub
    
    def extract_feat(self, inputs):
        pass  # Stub
    
    def encode_decode(self, inputs, batch_img_metas):
        pass  # Stub
    
    def loss(self, inputs, data_samples):
        """
        inputs: [B, 3, H, W]
        data_samples: list[SegDataSample]
        """
        device = inputs.device
        B = inputs.shape[0]
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        # 收集 gt；保留 255 不映射，让 CE 的 ignore_index=255 忽略
        gts = []
        for ds in data_samples:
            gt = ds.gt_sem_seg.data.squeeze(0).clone()  # [H, W]
            gts.append(gt)
        gt = torch.stack(gts).long().to(device)  # [B, H, W]，可含 255

        # [FIX] SAM3 expects 1024x1024 input due to pre-computed positional encodings
        # Resize inputs and gt to 1024x1024 for training
        original_size = (inputs.shape[2], inputs.shape[3])
        if original_size != (1024, 1024):
            inputs = F.interpolate(inputs, size=(1024, 1024), mode='bilinear', align_corners=False)
            gt = F.interpolate(gt.unsqueeze(1).float(), size=(1024, 1024), mode='nearest').squeeze(1).long()

        # Step 1: backbone forward_image (冻结)
        with torch.no_grad():
            backbone_out = self.processor.model.backbone.forward_image(inputs)  # 只图像

        # Step 2: 为所有 query_word 提前编码文本（batch 处理）
        all_txt_feats = []
        all_txt_masks = []
        for query_word in self.query_words:
            text_out = self.processor.model.backbone.forward_text([query_word] * B, device=device)
            all_txt_feats.append(text_out['language_features'])  # [B, seq_len_text, 256]
            all_txt_masks.append(text_out['language_mask'])      # [B, seq_len_text]
        # 堆叠成 [num_queries, B, seq_len, 256] 等

        fpn_feats = self._get_fpn(backbone_out)
        fpn_backup = [f.clone() for f in fpn_feats]

        seg_logits_list = []  # 收集每个 query 的 logit

        for q_i, query_word in enumerate(self.query_words):
            fpn_feats = [f.clone() for f in fpn_backup]

            # adapter 增强（每个 query 独立）
            # cached_prototypes 现在是 [topk, 512] 和 [topk]，adapter 会自动广播到 batch
            prototypes, proto_scores = self.cached_prototypes[query_word]

            enhanced_fpn = self.adapter(fpn_feats, prototypes, proto_scores)
            self._set_fpn(backbone_out, enhanced_fpn)

            # Step 3: 注入当前 query 的文本特征
            current_text_out = self.processor.model.backbone.forward_text([query_word] * B, device=device)
            backbone_out.update(current_text_out)  # 添加 language_features 等

            # forward_grounding
            outputs = self.processor.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self.processor.find_stage,
                geometric_prompt=self.processor.model._get_dummy_prompt(),
                find_target=None
            )

            # 提取并 compose logit（同推理逻辑）
            masks_logits = self._get_state_value(outputs, ("masks_logits", "pred_masks", "pred_masks_logits"))
            object_score = self._get_state_value(outputs, ("object_score", "pred_iou_scores", "iou_scores"))
            semantic_logits = self._get_state_value(outputs, ("semantic_mask_logits", "semantic_logits"))
            presence_score = self._get_state_value(outputs, ("presence_score", "pred_presence_score"))

            logit = self._compose_query_logit_baseline(
                masks_logits=masks_logits,
                object_score=object_score,
                semantic_logits=semantic_logits,
                presence_score=presence_score,
                out_hw=(gt.shape[1], gt.shape[2])
            )  # 返回 [B, H, W]

            seg_logits_list.append(logit.unsqueeze(1))  # [B, 1, H, W]

        # 聚合多词到类级别
        seg_logits = torch.cat(seg_logits_list, dim=1)  # [B, Q, H, W]
        if self.num_cls != self.num_queries:
            # build mapping mask: [Q, C] where mask[q, c] = 1 if query q -> class c
            q2c = F.one_hot(self.query_idx, num_classes=self.num_cls).to(seg_logits.dtype)  # [Q, C]
            # reshape to [1, Q, 1, 1, C] and seg_logits [B, Q, H, W] -> [B, Q, H, W, 1]
            # compute masked values and take max over Q for each class
            seg_exp = seg_logits.unsqueeze(-1)  # [B, Q, H, W, 1]
            mask = q2c.unsqueeze(0).unsqueeze(2).unsqueeze(2)  # [1, Q, 1, 1, C]
            masked = seg_exp * mask  # [B, Q, H, W, C]
            seg_by_class, _ = masked.max(dim=1)  # [B, H, W, C]
            seg_logits = seg_by_class.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        else:
            # queries already correspond to classes one-to-one
            seg_logits = seg_logits  # [B, C, H, W] if Q == C

        # seg_logits is already [B, num_cls, H, W], no need to expand

        # CE loss
        loss_ce = criterion(seg_logits, gt)

        # Dice loss（正确初始化）
        criterion_dice = DiceLoss(use_sigmoid=False, ignore_index=255)
        loss_dice = criterion_dice(seg_logits, gt)

        # 总 loss
        loss_seg = loss_ce + 0.5 * loss_dice  # 权重可调

        return {'loss_seg': loss_seg}

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices