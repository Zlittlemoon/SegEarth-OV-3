import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.models.losses import CrossEntropyLoss
from PIL import Image
from torchvision.transforms import v2

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ================= 新增 import =================
from adapter.adapter import DualSourceAdapter
from Offline_database.prototype_utils import PrototypeManager
# ==============================================


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
                 
                 # ================= 新增参数 =================
                 use_adapter=True,
                 clip_ckpt_path="weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
                 feature_db_path="Offline_database/Million-AID_train_image_features.pt",
                 # [新增] 接收你训练好的权重路径
                 adapter_ckpt_path=None,
                 # ===========================================
                 
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

        # [修改点 1] 初始化 Adapter 和 原型管理器
        # ============================================================
        self.use_adapter = use_adapter
        if self.use_adapter:
            # 1. 初始化 Adapter
            # clip_dim=512 (RemoteCLIP), sam_dim=256 (SAM Decoder Input)
            self.adapter = DualSourceAdapter(clip_dim=512, sam_dim=256).to(device)
            # [新增] 核心逻辑：如果提供了权重路径，就加载它！
            if adapter_ckpt_path is not None:
                print(f"[Adapter] Loading trained weights from: {adapter_ckpt_path}")
                # 加载权重 (map_location处理多卡到单卡的情况)
                state_dict = torch.load(adapter_ckpt_path, map_location=device)
                
                # 处理可能的 key 不匹配 (例如多卡训练时带 module. 前缀)
                # 因为你保存的是 model.module.adapter.state_dict()，所以 key 应该是干净的
                # 直接加载即可
                missing, unexpected = self.adapter.load_state_dict(state_dict, strict=False)
                print(f"[Adapter] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                print("[Adapter] No pretrained weights provided. Initializing randomly (Valid for Training only).")

            # 2. 初始化管理器 (加载大文件)
            self.proto_manager = PrototypeManager(
                clip_ckpt_path=clip_ckpt_path,
                database_path=feature_db_path,
                device=device
            )
            
            # 3. [预取策略] 提前把所有类别的原型检索好，存入缓存
            # 这样 inference 时不用重复跑 CLIP text encoder
            print("Pre-computing prototypes for all classes...")
            self.cached_prototypes = {} # 字典: class_word -> (v_ref, scores)
            
            # 批量检索 (Batch Retrieval)
            # v_refs: [Num_Cls, K, 512], scores: [Num_Cls, K]
            v_refs, scores = self.proto_manager.get_prototypes(self.query_words, topk=5)
            
            for i, word in enumerate(self.query_words):
                self.cached_prototypes[word] = (v_refs[i], scores[i])
            print("Prototypes ready.")
        # ============================================================        

    def _inference_single_view(self, image):
            w, h = image.size
            seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state = self.processor.set_image(image)

                # Robustness check for keys
                backbone_out = inference_state["backbone_out"]
                fpn_feats_list = None
                
                # [修复开始] ==========================================================
                # 1. 优先检查 sam2_backbone_out，且必须不为 None
                if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                    if "backbone_fpn" in backbone_out["sam2_backbone_out"]:
                        fpn_feats_list = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                
                # 2. 如果上面没取到，或者 sam2_backbone_out 是 None，尝试直接取 backbone_fpn
                if fpn_feats_list is None and "backbone_fpn" in backbone_out:
                    fpn_feats_list = backbone_out["backbone_fpn"]
                
                # 3. 如果还是 None，说明提取失败，抛出异常或跳过
                if fpn_feats_list is None:
                    # 打印一下当前的 keys 方便调试
                    raise RuntimeError(f"Could not find 'backbone_fpn'. Keys: {backbone_out.keys()}")
                # [修复结束] ==========================================================

                if self.use_adapter and fpn_feats_list is not None:
                    feat_backup = [f.clone() for f in fpn_feats_list]

                for query_idx, query_word in enumerate(self.query_words):
                    self.processor.reset_all_prompts(inference_state)

                    if self.use_adapter and fpn_feats_list is not None:
                        # Reset
                        for i, f in enumerate(feat_backup):
                            fpn_feats_list[i] = f.clone()

                        if query_word in self.cached_prototypes:
                            v_ref, v_scores = self.cached_prototypes[query_word]
                            v_ref_b = v_ref.unsqueeze(0)
                            v_scores_b = v_scores.unsqueeze(0)
                            
                            for i in range(len(fpn_feats_list)):
                                feat_map = fpn_feats_list[i]
                                B, C, H_f, W_f = feat_map.shape
                                
                                # Flatten Query
                                q_local = feat_map.flatten(2).transpose(1, 2)
                                
                                # [OOM Fix] Downsample Global Context (e.g., to 32x32)
                                # This reduces sequence length from 65536 to 1024
                                v_global_map = F.adaptive_avg_pool2d(feat_map, (32, 32)) 
                                v_global = v_global_map.flatten(2).transpose(1, 2)

                                enhanced_q = self.adapter(q_local, v_ref_b, v_scores_b, v_global)
                                fpn_feats_list[i] = enhanced_q.transpose(1, 2).view(B, C, H_f, W_f)

                    inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                    if self.use_transformer_decoder:
                        if inference_state['masks_logits'].shape[0] > 0:
                            inst_len = inference_state['masks_logits'].shape[0]
                            for inst_id in range(inst_len):
                                instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                                instance_score = inference_state['object_score'][inst_id]
                                
                                if instance_logits.shape != (h, w):
                                    instance_logits = F.interpolate(
                                        instance_logits.view(1, 1, *instance_logits.shape), 
                                        size=(h, w), mode='bilinear', align_corners=False
                                    ).squeeze()
                                seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                        
                        if self.use_sem_seg:
                            semantic_logits = inference_state['semantic_mask_logits']
                            if semantic_logits.shape != (h, w):
                                semantic_logits = F.interpolate(
                                    semantic_logits, size=(h, w), mode='bilinear', align_corners=False
                                ).squeeze()
                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                        
                        if self.use_presence_score:
                            seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                    
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
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples

    def _forward(self, image):
            """
            训练专用的前向传播 (修复 Mask 尺寸不匹配问题: 动态获取 H_mask, W_mask)
            """
            w, h = image.size 
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 1. Set Image (Batch=1)
                image_tensor = v2.functional.to_image(image).to(self.device)
                image_tensor = self.processor.transform(image_tensor).unsqueeze(0) 

                with torch.no_grad():
                    backbone_out = self.processor.model.backbone.forward_image(image_tensor)
                
                # --- FPN Feature Lookup ---
                fpn_feats_list = None
                if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                    if "backbone_fpn" in backbone_out["sam2_backbone_out"]:
                        fpn_feats_list = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                
                if fpn_feats_list is None and "backbone_fpn" in backbone_out:
                    fpn_feats_list = backbone_out["backbone_fpn"]

                if fpn_feats_list is None:
                    raise KeyError("Critical Error: Feature Pyramid not found.")

                # Inst Interactive Predictor Logic
                inst_interactivity_en = self.processor.model.inst_interactive_predictor is not None
                if inst_interactivity_en and "sam2_backbone_out" in backbone_out:
                    sam2_out = backbone_out["sam2_backbone_out"]
                    if sam2_out is not None and "backbone_fpn" in sam2_out:
                        if hasattr(self.processor.model.inst_interactive_predictor.model.sam_mask_decoder, 'conv_s0'):
                            with torch.no_grad():
                                sam2_out["backbone_fpn"][0] = (
                                    self.processor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                                        sam2_out["backbone_fpn"][0]
                                    )
                                )
                                sam2_out["backbone_fpn"][1] = (
                                    self.processor.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                                        sam2_out["backbone_fpn"][1]
                                    )
                                )

                inference_state = {
                    "original_height": h,
                    "original_width": w,
                    "backbone_out": backbone_out, 
                    "geometric_prompt": self.processor.model._get_dummy_prompt()
                }
                
                if self.use_adapter:
                    feat_backup = [f.clone() for f in fpn_feats_list]

                final_logits = torch.zeros((self.num_cls, h, w), device=self.device)
                final_logits.fill_(-100.0) 

                # Loop Classes
                for query_idx, query_word in enumerate(self.query_words):
                    if "backbone_out" in inference_state:
                        for key in ["language_features", "language_mask", "language_embeds"]:
                            inference_state["backbone_out"].pop(key, None)
                    for key in ["boxes", "masks", "masks_logits", "scores"]:
                        inference_state.pop(key, None)

                    # --- Adapter Injection ---
                    if self.use_adapter:
                        for i, f in enumerate(feat_backup):
                            fpn_feats_list[i] = f.clone()
                        
                        if query_word in self.cached_prototypes:
                            v_ref, v_scores = self.cached_prototypes[query_word]
                            v_ref_b = v_ref.unsqueeze(0)   
                            v_scores_b = v_scores.unsqueeze(0) 

                            for i in range(len(fpn_feats_list)):
                                feat_map = fpn_feats_list[i]
                                B, C, H_f, W_f = feat_map.shape
                                q_local = feat_map.flatten(2).transpose(1, 2)
                                
                                v_global_map = F.adaptive_avg_pool2d(feat_map, (32, 32)) 
                                v_global = v_global_map.flatten(2).transpose(1, 2)

                                enhanced_q = self.adapter(q_local, v_ref_b, v_scores_b, v_global)
                                fpn_feats_list[i] = enhanced_q.transpose(1, 2).view(B, C, H_f, W_f)

                    with torch.no_grad():
                        text_outputs = self.processor.model.backbone.forward_text([query_word], device=self.device)
                    inference_state["backbone_out"].update(text_outputs)

                    outputs = self.processor.model.forward_grounding(
                        backbone_out=inference_state["backbone_out"],
                        find_input=self.processor.find_stage,
                        geometric_prompt=inference_state["geometric_prompt"],
                        find_target=None,
                    )

                    # ================================================================
                    # [核心修复] 动态获取 Mask 尺寸 + Batch 处理
                    # ================================================================
                    out_logits = outputs["pred_logits"] 
                    out_masks = outputs["pred_masks"]   
                    presence_logit = outputs["presence_logit_dec"]

                    # 1. 强制移除 Batch 维度
                    if out_logits.shape[0] == 1:
                        out_logits = out_logits.squeeze(0) # [N] or [N, 1]
                    
                    if out_masks.shape[0] == 1:
                        out_masks = out_masks.squeeze(0)   # [N, 1, H_mask, W_mask]

                    if out_masks.ndim == 4 and out_masks.shape[1] == 1:
                        out_masks = out_masks.squeeze(1)   # [N, H_mask, W_mask]

                    N_inst = out_masks.shape[0]
                    
                    # [关键修复] 动态获取 Mask 高宽，不再写死 256
                    H_mask, W_mask = out_masks.shape[-2:] 
                    
                    if N_inst > 0:
                        obj_scores_all = out_logits.sigmoid().view(-1, 1, 1)
                        
                        # 使用动态获取的尺寸初始化
                        running_max_prob = torch.zeros((H_mask, W_mask), device=self.device, dtype=torch.float32)
                        CHUNK_SIZE = 512
                        
                        for start in range(0, N_inst, CHUNK_SIZE):
                            end = min(start + CHUNK_SIZE, N_inst)
                            
                            chunk_masks = out_masks[start:end].sigmoid().to(torch.float32)
                            chunk_scores = obj_scores_all[start:end].to(torch.float32)
                            
                            chunk_probs = chunk_masks * chunk_scores 
                            chunk_max, _ = chunk_probs.max(dim=0)
                            running_max_prob = torch.max(running_max_prob, chunk_max)
                            
                            del chunk_masks, chunk_scores, chunk_probs, chunk_max
                        
                        if self.use_presence_score:
                            presence_prob = presence_logit.sigmoid().to(torch.float32)
                            if presence_prob.numel() > 1:
                                presence_prob = presence_prob.mean()
                            running_max_prob = running_max_prob * presence_prob

                        final_cls_prob = F.interpolate(
                            running_max_prob.unsqueeze(0).unsqueeze(0), 
                            size=(h, w), 
                            mode="bilinear", 
                            align_corners=False
                        ).squeeze()

                        final_cls_prob = torch.clamp(final_cls_prob, 1e-6, 1-1e-6)
                        final_cls_logit = torch.logit(final_cls_prob)
                        final_logits[query_idx] = final_cls_logit
                    
                return final_logits
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
            """
            显存优化版 Loss 计算：Iterative Backward
            """
            # 获取 image 和 gt
            img_path = data_samples[0].metainfo["img_path"]
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            
            device = inputs.device
            gt = data_samples[0].gt_sem_seg.data.squeeze(0).to(device) # [H, W]

            # ==========================================================
            # 1. 预处理图像和 Backbone (这一步只做一次，不费显存)
            # ==========================================================
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_tensor = v2.functional.to_image(image).to(device)
                image_tensor = self.processor.transform(image_tensor).unsqueeze(0)
                backbone_out = self.processor.model.backbone.forward_image(image_tensor)

                # 提取 FPN (逻辑同你原代码)
                fpn_feats_list = None
                # 1. 优先尝试从 sam2_backbone_out 获取，并确保它不是 None
                if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                    if "backbone_fpn" in backbone_out["sam2_backbone_out"]:
                        fpn_feats_list = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                
                # 2. 如果上面没拿到，尝试直接从 backbone_out 获取
                if fpn_feats_list is None and "backbone_fpn" in backbone_out:
                    fpn_feats_list = backbone_out["backbone_fpn"]

                # 3. 终极检查：如果没有拿到特征，抛出清晰的错误
                if fpn_feats_list is None:
                    available_keys = list(backbone_out.keys())
                    raise RuntimeError(f"Cannot find 'backbone_fpn' in output. Available keys: {available_keys}")
                
                feat_backup = [f.clone() for f in fpn_feats_list]

            # ==========================================================
            # 2. 循环计算 Loss 并即时 Backward (O(1) 显存核心)
            # ==========================================================
            total_loss_value = 0.0 # 用于记录数值打印，不用于梯度
            
            # 使用 BCEWithLogitsLoss 代替 CE Loss
            # 针对每个像素：是该类(1) 还是 不是该类(0)
            criterion = nn.BCEWithLogitsLoss() 

            for query_idx, query_word in enumerate(self.query_words):
                # 2.1 准备 GT：将多分类 GT 转为当前类别的二分类 Mask (0 或 1)
                # 忽略 255 区域
                valid_mask = (gt != 255)
                target_mask = torch.zeros_like(gt, dtype=torch.float32)
                target_mask[valid_mask & (gt == query_idx)] = 1.0
                target_mask[valid_mask & (gt != query_idx)] = 0.0
                
                # 如果该类别在图中完全不存在，且你希望节省时间，可以跳过 (可选)
                # if target_mask.sum() == 0: continue 

                # 开启混合精度和梯度记录
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # --- 重置特征 ---
                    # 每次迭代必需使用原始特征的拷贝，否则特征会被 Adapter 越改越乱
                    current_fpn = [f.clone() for f in feat_backup] 

                    # --- 准备 Prompt 状态 ---
                    inference_state = {
                        "original_height": h, "original_width": w,
                        "backbone_out": backbone_out.copy(), # 浅拷贝字典
                        "geometric_prompt": self.processor.model._get_dummy_prompt()
                    }

                    # --- Adapter (有梯度!) ---
                    if self.use_adapter:
                        # 检索 Prototype (缓存)
                        if query_word in self.cached_prototypes:
                            v_ref, v_scores = self.cached_prototypes[query_word]
                            v_ref_b, v_scores_b = v_ref.unsqueeze(0), v_scores.unsqueeze(0)
                            
                            for i in range(len(current_fpn)):
                                feat_map = current_fpn[i]
                                B_f, C_f, H_f, W_f = feat_map.shape
                                q_local = feat_map.flatten(2).transpose(1, 2)
                                
                                # 全局池化减少显存
                                v_global = F.adaptive_avg_pool2d(feat_map, (32, 32)).flatten(2).transpose(1, 2)
                                
                                # Adapter 前向
                                enhanced_q = self.adapter(q_local, v_ref_b, v_scores_b, v_global)
                                current_fpn[i] = enhanced_q.transpose(1, 2).view(B_f, C_f, H_f, W_f)
                    
                    # --- SAM Decoder (冻结，但需要传递 Adapter 的梯度) ---
                    # 注意：这里我们临时把 current_fpn 塞回 inference_state
                    # 这一步依赖具体 SAM3 实现，假设它是引用传递
                    # [修复] 塞回特征时，必须检查 sam2_backbone_out 是否为 None
                    if "sam2_backbone_out" in inference_state["backbone_out"] and \
                    inference_state["backbone_out"]["sam2_backbone_out"] is not None:
                        inference_state["backbone_out"]["sam2_backbone_out"]["backbone_fpn"] = current_fpn
                    else:
                        # 如果 sam2_backbone_out 是 None，说明特征是从外层 backbone_fpn 取的
                        # 所以我们要塞回外层
                        inference_state["backbone_out"]["backbone_fpn"] = current_fpn

                    # 获取 Text Feature
                    with torch.no_grad():
                        text_outputs = self.processor.model.backbone.forward_text([query_word], device=device)
                    inference_state["backbone_out"].update(text_outputs)

                    # Decoder 前向
                    outputs = self.processor.model.forward_grounding(
                        backbone_out=inference_state["backbone_out"],
                        find_input=self.processor.find_stage,
                        geometric_prompt=inference_state["geometric_prompt"],
                        find_target=None  # <--- ✅ 补上这个参数
                    )

                    # --- 后处理 Logits ---
                    # 这里只取第0个 query 的结果 (因为我们只输出了单类)
                    # SAM3 输出通常是 [Batch, Queries, H, W] -> [1, 1, H, W]
                    pred_logits = outputs["pred_logits"] 
                    
                    # 插值对齐 GT
                    # [DEBUG] 打印形状（调试完可注释掉）
                    # print(f"Raw logits shape: {pred_logits.shape}")

                    # [修复] interpolate 要求输入必须是 4D (B, C, H, W)
                    # 如果当前是 3D (例如 [1, H, W])，需要 unsqueeze 出 Channel 维度
                    if pred_logits.ndim == 3:
                        pred_logits = pred_logits.unsqueeze(1) # 变成 [1, 1, H, W]

                    # 插值对齐 GT
                    if pred_logits.shape[-2:] != gt.shape:
                        pred_logits = F.interpolate(
                            pred_logits, 
                            size=gt.shape, 
                            mode="bilinear", 
                            align_corners=False
                        )
                    
                    # 恢复成 2D [H, W] 用于计算 Loss
                    pred_logits = pred_logits.squeeze() 

                    # --- 计算单类 Loss ---
                    # 忽略 255 区域
                    # target_mask 已经在循环开头定义好了
                    
                    # [新增] 关键检查：如果这张图全是 255，或者当前类别区域全是 255
                    # 有效像素数 = 0 -> 导致 BCE Loss = NaN
                    if valid_mask.sum() == 0:
                        continue  # 直接跳过，不计算 Loss，也不 Backward

                    # 只在有效区域计算
                    loss_i = criterion(pred_logits[valid_mask], target_mask[valid_mask])

                    # 防止 squeeze 过度（例如 Batch=1, H=512, W=512 被压成 [512, 512]，这是对的）
                    # 但如果刚好 H 或 W 是 1，可能会出错。加个保险：
                    if pred_logits.ndim == 3: # 如果还是 [1, 512, 512]
                        pred_logits = pred_logits.squeeze(0)

                    # --- 计算单类 Loss ---
                    # 只在有效区域计算 (排除 255)
                    loss_i = criterion(pred_logits[valid_mask], target_mask[valid_mask])
                
                # ==========================================================
                # 3. 立即反向传播并释放显存 (Critical Step)
                # ==========================================================
                # 注意：因为我们在外部循环累积梯度，所以如果是 Batch>1，需要除以累积步数
                # 但这里我们已经在内部解决了，直接 Backward
                
                # 这里的 loss_i 是计算图的一部分，backward 会把梯度传回 Adapter
                loss_i.backward() 
                
                total_loss_value += loss_i.item()

                # 手动删除计算图起点，断开连接，释放显存
                del pred_logits, loss_i, current_fpn, inference_state, outputs
            
            # 返回平均 Loss 用于日志记录 (此时只是一个 float，没有梯度了)
            return dict(loss_seg=total_loss_value / len(self.query_words))

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
