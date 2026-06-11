import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model, build_rs_prompt_semantic_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate


@MODELS.register_module()
class SegEarthOV3SegmentationSoftPrompt(BaseSegmentor):
    """
    Soft-prompt compatible version of segearthov3_segmentor_merge.py.

    Design goal:
    - Keep category merge and multi-query merge logic identical to the original merge file.
    - Only swap model/weight loading path to optionally use soft-prompt fine-tuned checkpoints.
    """

    def __init__(
        self,
        classname_path,
        device=torch.device("cuda"),
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        checkpoint_path="weights/sam3/sam3.pt",
        finetuned_checkpoint_path=None,
        use_soft_prompt=False,
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        **kwargs,
    ):
        super().__init__()

        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

        self.use_soft_prompt = bool(use_soft_prompt) and finetuned_checkpoint_path is not None

        if self.use_soft_prompt:
            # Build wrapper model used by soft-prompt training.
            model = build_rs_prompt_semantic_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device="cpu",
                eval_mode=True,
                enable_segmentation=True,
            )
            self._load_finetuned_state_dict(model, finetuned_checkpoint_path)
            model = model.to(self.device).eval()
            self.model = model
            self.sam3_core = model.sam3
            self.use_sr = bool(getattr(model, "use_sr", False))
            self.soft_prompt_pos = getattr(model, "soft_prompt_pos", "post_concat")

            # Template find_input for single-query inference.
            self.soft_find_input = FindStage(
                img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
        else:
            # Original SAM3 path.
            model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device="cpu",
                eval_mode=True,
            )
            model = model.to(self.device).eval()
            self.model = model
            self.sam3_core = model
            self.soft_find_input = None
            self.use_sr = False
            self.soft_prompt_pos = "none"

        # Keep original processor-driven inference path for image prep and baseline model path.
        self.processor = Sam3Processor(
            self.sam3_core,
            confidence_threshold=confidence_threshold,
            device=self.device,
        )

        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(self.device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        if self.use_soft_prompt:
            print(
                "[SOFT PROMPT INFER] "
                f"soft_prompt_pos={self.soft_prompt_pos}, use_sr={self.use_sr}"
            )

    def _load_finetuned_state_dict(self, model, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        cleaned = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v

        model_state = model.state_dict()
        resized_keys = []
        skipped_shape_keys = []

        def _resize_tensor_to_shape(src: torch.Tensor, dst_shape):
            out = src.new_zeros(dst_shape)
            common_slices = tuple(slice(0, min(int(s), int(d))) for s, d in zip(src.shape, dst_shape))
            out[common_slices] = src[common_slices]
            return out

        for k in list(cleaned.keys()):
            if k not in model_state:
                continue
            ckpt_v = cleaned[k]
            model_v = model_state[k]
            if not torch.is_tensor(ckpt_v) or not torch.is_tensor(model_v):
                continue
            if tuple(ckpt_v.shape) == tuple(model_v.shape):
                continue

            # Soft prompt length can differ between training and inference envs.
            # Keep overlap and pad remaining entries with current model init.
            if k == "soft_prompt" and ckpt_v.ndim == model_v.ndim:
                resized = _resize_tensor_to_shape(ckpt_v, model_v.shape).to(
                    dtype=model_v.dtype
                )
                cleaned[k] = resized
                resized_keys.append(
                    f"{k}: ckpt{tuple(ckpt_v.shape)} -> model{tuple(model_v.shape)}"
                )
                continue

            # For other mismatched tensors, skip loading that key.
            skipped_shape_keys.append(
                f"{k}: ckpt{tuple(ckpt_v.shape)} vs model{tuple(model_v.shape)}"
            )
            cleaned.pop(k, None)

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        ckpt_has_sr_adapter = any(k.startswith("sr_adapter.") for k in cleaned.keys())
        model_has_sr_adapter = hasattr(model, "sr_adapter") and model.sr_adapter is not None
        print(f"[SOFT PROMPT LOAD] checkpoint={ckpt_path}")
        print(f"[SOFT PROMPT LOAD] missing keys   : {len(missing)}")
        print(f"[SOFT PROMPT LOAD] unexpected keys: {len(unexpected)}")
        if ckpt_has_sr_adapter and not model_has_sr_adapter:
            print(
                "[SOFT PROMPT LOAD][WARNING] checkpoint contains sr_adapter.* "
                "but the inference model did not instantiate sr_adapter. "
                "Set SAM3_RS_USE_SR=1 and the SAM3_RS_SR_* env vars before building "
                "the model if this is an SR-adapter checkpoint."
            )
        if len(resized_keys) > 0:
            print(f"[SOFT PROMPT LOAD] resized keys  : {len(resized_keys)}")
            print("[SOFT PROMPT LOAD] first resized :", resized_keys[:20])
        if len(skipped_shape_keys) > 0:
            print(f"[SOFT PROMPT LOAD] skipped shape : {len(skipped_shape_keys)}")
            print("[SOFT PROMPT LOAD] first skipped :", skipped_shape_keys[:20])
        if len(missing) > 0:
            print("[SOFT PROMPT LOAD] first missing:", missing[:20])
        if len(unexpected) > 0:
            print("[SOFT PROMPT LOAD] first unexpected:", unexpected[:20])

    @staticmethod
    def _extract_first_tensor(value, field_name: str):
        """Accept Tensor or nested tuple/list and return the first Tensor found."""
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = SegEarthOV3SegmentationSoftPrompt._extract_first_tensor(item, field_name)
                if tensor is not None:
                    return tensor
        raise TypeError(
            f"{field_name} is expected to be a Tensor (or container with Tensor), "
            f"but got type={type(value)}"
        )

    def _image_to_model_tensor(self, image):
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            height, width = image.shape[-2:]

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.processor.transform(image_tensor).unsqueeze(0)
        return image_tensor, height, width

    def _set_image_soft_prompt(self, image):
        image_tensor, height, width = self._image_to_model_tensor(image)
        backbone_out = self.sam3_core.backbone.forward_image(image_tensor)
        if self.use_sr:
            backbone_out = self.model._apply_sr_to_backbone_out(backbone_out, image_tensor)
        return {
            "original_height": height,
            "original_width": width,
            "backbone_out": backbone_out,
        }

    def _forward_text_soft_prompt(self, query_word: str):
        pre_text_soft_prompt = None
        if self.soft_prompt_pos == "pre_text":
            pre_text_soft_prompt = self.model._build_pre_text_soft_prompt(batch_size=1)
            if pre_text_soft_prompt is not None:
                pre_text_soft_prompt = pre_text_soft_prompt.to(device=self.device)

        text_outputs = self.sam3_core.backbone.forward_text(
            [query_word],
            device=self.device,
            soft_prompt_embed=pre_text_soft_prompt,
        )

        if self.soft_prompt_pos == "post_text_add":
            text_outputs = self.model._apply_post_text_soft_prompt(text_outputs)

        return text_outputs

    def _forward_query_soft_prompt(self, inference_state, query_word: str, h: int, w: int):
        """
        Run one query with soft-prompt tuning active and return a state dict
        compatible with the original inference loop keys.
        """
        # 1) update text features on top of cached image backbone features
        text_outputs = self._forward_text_soft_prompt(query_word)
        backbone_out = inference_state["backbone_out"]
        backbone_out.update(text_outputs)

        # 2) encode prompt + apply soft prompt
        find_input = self.soft_find_input
        geometric_prompt = self.model._build_geometric_prompt(find_input)
        prompt, prompt_mask, backbone_out = self.sam3_core._encode_prompt(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=None,
            visual_prompt_mask=None,
            encode_text=True,
            prev_mask_pred=None,
        )
        if self.soft_prompt_pos == "post_concat":
            tuned_prompt = self.model._apply_prompt_tuning(prompt, prompt_mask)
            if isinstance(tuned_prompt, tuple) and len(tuned_prompt) == 2:
                prompt, prompt_mask = tuned_prompt
            else:
                # Backward compatibility if helper returns prompt only.
                prompt = tuned_prompt

        # Compatibility: some codepaths can wrap prompt/prompt_mask in tuple/list.
        prompt = self._extract_first_tensor(prompt, "prompt")
        prompt_mask = self._extract_first_tensor(prompt_mask, "prompt_mask")

        prompt = prompt.to(self.device)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(self.device)

        # 3) encoder/decoder/heads
        backbone_out, encoder_out, _ = self.sam3_core._run_encoder(
            backbone_out=backbone_out,
            find_input=find_input,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_extra_kwargs=None,
        )

        out = {}
        out, hs = self.sam3_core._run_decoder(
            pos_embed=encoder_out["pos_embed"],
            memory=encoder_out["encoder_hidden_states"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=encoder_out["prompt_after_enc"],
            prompt_mask=encoder_out["prompt_mask"],
            encoder_out=encoder_out,
        )

        self.sam3_core._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=encoder_out["prompt_after_enc"],
            prompt_mask=encoder_out["prompt_mask"],
            hs=hs,
        )

        # 4) follow Sam3Processor._forward_grounding post-process semantics
        out_logits = out["pred_logits"]
        out_masks = out["pred_masks"]

        presence_score = out["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_logits.sigmoid() * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]

        if out_masks.numel() > 0:
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (h, w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid().squeeze(1)
        else:
            out_masks = out_masks.new_zeros((0, h, w))

        out_semantic = interpolate(
            out["semantic_seg"],
            (h, w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid().squeeze()

        return {
            "masks_logits": out_masks,
            "object_score": out_probs,
            "semantic_mask_logits": out_semantic,
            "presence_score": out["presence_logit_dec"].sigmoid().squeeze(),
            "object_score_includes_presence": True,
            "semantic_score_includes_presence": False,
        }

    @staticmethod
    def _score_without_presence(score, presence_score):
        if torch.is_tensor(presence_score):
            return score / presence_score.to(score.device).clamp_min(1.0e-6)
        return score / max(float(presence_score), 1.0e-6)

    def _maybe_apply_presence_to_instance_score(self, query_state, instance_score):
        presence_score = query_state.get("presence_score", None)
        includes_presence = bool(query_state.get("object_score_includes_presence", True))

        if presence_score is None:
            return instance_score
        if includes_presence and not self.use_presence_score:
            return self._score_without_presence(instance_score, presence_score)
        if (not includes_presence) and self.use_presence_score:
            return instance_score * presence_score
        return instance_score

    def _maybe_apply_presence_to_semantic_score(self, query_state, semantic_logits):
        presence_score = query_state.get("presence_score", None)
        includes_presence = bool(query_state.get("semantic_score_includes_presence", False))

        if presence_score is None:
            return semantic_logits
        if includes_presence and not self.use_presence_score:
            return self._score_without_presence(semantic_logits, presence_score)
        if (not includes_presence) and self.use_presence_score:
            return semantic_logits * presence_score
        return semantic_logits

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        autocast_enabled = self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            if self.use_soft_prompt:
                inference_state = self._set_image_soft_prompt(image)
            else:
                inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                if self.use_soft_prompt:
                    # Soft-prompt path (only model forward differs).
                    query_state = self._forward_query_soft_prompt(inference_state, query_word, h, w)
                else:
                    # Original path.
                    self.processor.reset_all_prompts(inference_state)
                    query_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if query_state["masks_logits"].shape[0] > 0:
                        inst_len = query_state["masks_logits"].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = query_state["masks_logits"][inst_id].squeeze()
                            instance_score = query_state["object_score"][inst_id]
                            instance_score = self._maybe_apply_presence_to_instance_score(
                                query_state,
                                instance_score,
                            )

                            # Keep same compatibility guard as original implementation.
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape),
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(
                                seg_logits[query_idx],
                                instance_logits * instance_score,
                            )

                if self.use_sem_seg:
                    semantic_logits = query_state["semantic_mask_logits"]
                    if semantic_logits.shape != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits,
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()

                    semantic_logits = self._maybe_apply_presence_to_semantic_score(
                        query_state,
                        semantic_logits,
                    )
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)

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
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get("img_path")
            image = Image.open(image_path).convert("RGB")
            ori_shape = meta["ori_shape"]

            # Determine inference mode
            if self.slide_crop > 0 and (
                self.slide_crop < image.size[0] or self.slide_crop < image.size[1]
            ):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Post-processing (kept identical to merge baseline)
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

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
                }
            )

        return data_samples

    def _forward(data_samples):
        """
        """

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
        """


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(",")
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "") for item in class_names]
    return class_names, class_indices
