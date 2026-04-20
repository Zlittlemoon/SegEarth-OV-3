from __future__ import annotations

import contextlib
import os
from typing import Any, Dict

import torch
import torch.nn as nn

from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image import Sam3Image


class RSPromptSemanticModel(nn.Module):
    """
    Prompt-conditioned semantic wrapper for SAM3 with optional prompt-only tuning.

    Supports batch > 1:
      - batch.img_batch: [B, 3, H, W]
      - batch.find_text_batch: list[str] of length B
      - find_input.text_ids / img_ids: tensors of length B
      - prompt: [L, B, C]
      - semantic_seg: [B, 1, H, W]
    """

    def __init__(self, sam3_model: Sam3Image):
        super().__init__()
        self.sam3 = sam3_model
        self.debug = os.environ.get("SAM3_RS_DEBUG_PROMPT", "0") == "1"
        self.debug_first_n = int(os.environ.get("SAM3_RS_DEBUG_FIRST_N", "2"))
        self.debug_every = int(os.environ.get("SAM3_RS_DEBUG_EVERY", "100"))
        self._forward_calls = 0
        self._debug_this_step = False

        if getattr(self.sam3, "segmentation_head", None) is None:
            raise ValueError(
                "RSPromptSemanticModel requires sam3_model.segmentation_head to be enabled."
            )

        # Prompt-tuning config
        self.prompt_tuning_mode = os.environ.get("SAM3_RS_PROMPT_TUNING", "soft").strip().lower()
        self.soft_prompt_len = int(os.environ.get("SAM3_RS_SOFT_PROMPT_LEN", "34"))
        self.prompt_dim = int(os.environ.get("SAM3_RS_PROMPT_DIM", "256"))
        self.soft_prompt_pos = os.environ.get("SAM3_RS_SOFT_PROMPT_POS", "post_concat").strip().lower()
        if self.soft_prompt_pos not in {"post_concat", "pre_text", "post_text_add"}:
            raise ValueError(
                f"Unknown SAM3_RS_SOFT_PROMPT_POS={self.soft_prompt_pos}. "
                "Expected 'post_concat', 'pre_text', or 'post_text_add'."
            )
        # Default to post mode soft prompt (residual add on prompt tokens) to preserve text conditioning.
        self.prompt_only = os.environ.get("SAM3_RS_PROMPT_ONLY", "0") == "1"

        if self.prompt_tuning_mode == "soft":
            self.soft_prompt = nn.Parameter(
                torch.zeros(self.soft_prompt_len, 1, self.prompt_dim)
            )
            nn.init.normal_(self.soft_prompt, std=0.02)
            text_width = int(getattr(self.sam3.backbone.language_backbone.encoder, "width", self.prompt_dim))
            post_text_width = int(
                getattr(getattr(self.sam3.backbone.language_backbone, "resizer", None), "out_features", self.prompt_dim)
            )
            if self.soft_prompt_pos == "pre_text":
                target_width = text_width
            elif self.soft_prompt_pos == "post_text_add":
                target_width = post_text_width
            else:
                target_width = self.prompt_dim
            if target_width != self.prompt_dim:
                self.soft_prompt_proj = nn.Linear(self.prompt_dim, target_width, bias=False)
            else:
                self.soft_prompt_proj = None

        elif self.prompt_tuning_mode == "mlp":
            if self.soft_prompt_pos == "pre_text":
                raise ValueError("SAM3_RS_SOFT_PROMPT_POS='pre_text' requires SAM3_RS_PROMPT_TUNING='soft'.")
            self.prompt_mlp = nn.Sequential(
                nn.Linear(self.prompt_dim, self.prompt_dim),
                nn.GELU(),
                nn.Linear(self.prompt_dim, self.prompt_dim),
            )

        else:
            raise ValueError(
                f"Unknown SAM3_RS_PROMPT_TUNING={self.prompt_tuning_mode}. "
                f"Expected 'soft' or 'mlp'."
            )

        if self.prompt_only:
            self.freeze_sam3_and_train_prompt_only()

    @property
    def device(self):
        return self.sam3.device

    def back_convert(self, x):
        return self.sam3.back_convert(x)

    def on_validation_epoch_start(self):
        if hasattr(self.sam3, "on_validation_epoch_start"):
            return self.sam3.on_validation_epoch_start()

    def on_validation_epoch_end(self):
        if hasattr(self.sam3, "on_validation_epoch_end"):
            return self.sam3.on_validation_epoch_end()

    def _dbg(self, *args, **kwargs):
        if self.debug and self._debug_this_step:
            print(*args, **kwargs)

    def _decide_debug_step(self):
        self._forward_calls += 1
        step = self._forward_calls
        if not self.debug:
            self._debug_this_step = False
            return
        if step <= self.debug_first_n:
            self._debug_this_step = True
        elif self.debug_every > 0 and step % self.debug_every == 0:
            self._debug_this_step = True
        else:
            self._debug_this_step = False
        if self._debug_this_step:
            mode = "train" if self.training else "eval"
            print(
                f"[RSPromptSemanticModel][DEBUG] forward_step={step}, "
                f"mode={mode}, debug_first_n={self.debug_first_n}, "
                f"debug_every={self.debug_every}"
            )

    def freeze_sam3_and_train_prompt_only(self):
        for p in self.sam3.parameters():
            p.requires_grad = False

        if hasattr(self, "soft_prompt"):
            self.soft_prompt.requires_grad = True

        if hasattr(self, "soft_prompt_proj") and self.soft_prompt_proj is not None:
            for p in self.soft_prompt_proj.parameters():
                p.requires_grad = True

        if hasattr(self, "prompt_mlp"):
            for p in self.prompt_mlp.parameters():
                p.requires_grad = True

    def _get_find_input(self, batch: Any):
        """
        Return the stage-0 find_input, but do NOT assume only one prompt inside it.
        """
        if hasattr(batch, "find_inputs"):
            find_inputs = batch.find_inputs
            if isinstance(find_inputs, (list, tuple)) and len(find_inputs) > 0:
                return find_inputs[0]

        if hasattr(batch, "find_inputs_per_stage"):
            fps = batch.find_inputs_per_stage
            if isinstance(fps, (list, tuple)) and len(fps) > 0:
                stage0 = fps[0]
                if isinstance(stage0, (list, tuple)) and len(stage0) > 0:
                    return stage0[0]
                return stage0

        if hasattr(batch, "find_input"):
            return batch.find_input

        raise RuntimeError(
            "Could not find a prompt input on batch. Expected one of: "
            "batch.find_inputs / batch.find_inputs_per_stage / batch.find_input"
        )

    def _get_captions(self, batch: Any):
        if hasattr(batch, "find_text_batch"):
            return batch.find_text_batch
        if hasattr(batch, "captions"):
            return batch.captions
        raise RuntimeError(
            "Could not find captions on batch. Expected batch.find_text_batch or batch.captions"
        )

    def _get_images(self, batch: Any):
        if hasattr(batch, "img_batch"):
            return batch.img_batch
        if hasattr(batch, "images"):
            return batch.images
        raise RuntimeError(
            "Could not find image tensor on batch. Expected batch.img_batch or batch.images"
        )

    def _check_batched_prompt_input(self, find_input: Any, batch: Any):
        if not hasattr(find_input, "text_ids"):
            raise RuntimeError("find_input.text_ids is missing; text prompt was not passed in.")
        if not hasattr(find_input, "img_ids"):
            raise RuntimeError("find_input.img_ids is missing; image indices were not passed in.")

        text_ids = find_input.text_ids
        img_ids = find_input.img_ids

        if not torch.is_tensor(text_ids):
            raise RuntimeError(f"find_input.text_ids must be a tensor, got {type(text_ids)}")
        if not torch.is_tensor(img_ids):
            raise RuntimeError(f"find_input.img_ids must be a tensor, got {type(img_ids)}")

        num_prompts = int(text_ids.numel())
        num_img_refs = int(img_ids.numel())

        if num_prompts != num_img_refs:
            raise RuntimeError(
                f"Mismatched prompt/image counts inside find_input: "
                f"num_prompts={num_prompts}, num_img_refs={num_img_refs}, "
                f"text_ids={text_ids}, img_ids={img_ids}"
            )

        captions = self._get_captions(batch)
        images = self._get_images(batch)

        if not torch.is_tensor(images):
            raise RuntimeError(f"batch images must be a tensor, got {type(images)}")

        batch_imgs = int(images.shape[0])

        # img_ids indexes into batch.img_batch
        if num_img_refs > 0:
            min_img_id = int(img_ids.min().item())
            max_img_id = int(img_ids.max().item())
            if min_img_id < 0 or max_img_id >= batch_imgs:
                raise RuntimeError(
                    f"find_input.img_ids out of range: min={min_img_id}, max={max_img_id}, "
                    f"but batch image size is {batch_imgs}"
                )

        # text_ids indexes into captions table; captions may be deduplicated,
        # so len(captions) does NOT need to equal num_prompts.
        if len(captions) == 0:
            raise RuntimeError("captions table is empty")

        if num_prompts > 0:
            min_text_id = int(text_ids.min().item())
            max_text_id = int(text_ids.max().item())
            if min_text_id < 0 or max_text_id >= len(captions):
                raise RuntimeError(
                    f"find_input.text_ids out of range: min={min_text_id}, max={max_text_id}, "
                    f"but captions table has size {len(captions)}"
                )

    def _build_geometric_prompt(self, find_input):
        """
        Match official Sam3Image.forward():
          Prompt(
              box_embeddings=find_input.input_boxes,
              box_mask=find_input.input_boxes_mask,
              box_labels=find_input.input_boxes_label,
          )
        """
        boxes = getattr(find_input, "input_boxes", None)
        box_mask = getattr(find_input, "input_boxes_mask", None)
        box_labels = getattr(find_input, "input_boxes_label", None)

        has_boxes = boxes is not None and boxes.numel() > 0

        if has_boxes:
            return Prompt(
                box_embeddings=boxes,
                box_mask=box_mask,
                box_labels=box_labels,
            )

        num_prompts = 1
        if hasattr(find_input, "img_ids") and find_input.img_ids is not None:
            num_prompts = int(find_input.img_ids.numel())

        return self.sam3._get_dummy_prompt(num_prompts=num_prompts)

    def _apply_prompt_tuning(self, prompt, prompt_mask=None):
        soft_prompt_param = getattr(self, "soft_prompt", None)
        if soft_prompt_param is None:
            return prompt, prompt_mask
        if self.soft_prompt_pos != "post_concat":
            return prompt, prompt_mask

        if prompt.shape[-1] != self.prompt_dim:
            raise RuntimeError(
                f"Soft prompt dim mismatch: prompt.shape={tuple(prompt.shape)}, "
                f"expected last dim={self.prompt_dim}"
            )

        prompt_len = prompt.shape[0]
        batch_size = prompt.shape[1]
        soft_len = soft_prompt_param.shape[0]
        if soft_len < prompt_len:
            raise RuntimeError(
                f"Soft prompt length mismatch: prompt.shape={tuple(prompt.shape)}, "
                f"soft_prompt.shape={tuple(soft_prompt_param.shape)}. "
                f"Need soft_len >= prompt_len for residual add."
            )

        # Match token length for residual add: [L_prompt, 1, C]
        soft_prompt = soft_prompt_param[:prompt_len]

        # Expand to current batch: [L_prompt, B, C]
        if soft_prompt.shape[1] != batch_size:
            soft_prompt = soft_prompt.expand(-1, batch_size, -1)

        if self.prompt_only:
            tuned_prompt = soft_prompt
            if prompt_mask is None:
                tuned_mask = None
            else:
                tuned_mask = torch.zeros(
                    (batch_size, prompt_len),
                    device=prompt_mask.device,
                    dtype=prompt_mask.dtype,
                )
            return tuned_prompt, tuned_mask

        # Residual add on prompt tokens.
        tuned_prompt = prompt + soft_prompt
        return tuned_prompt, prompt_mask

    def _build_pre_text_soft_prompt(self, batch_size: int):
        soft_prompt_param = getattr(self, "soft_prompt", None)
        if soft_prompt_param is None:
            return None
        soft_prompt = soft_prompt_param
        if soft_prompt.shape[1] != batch_size:
            soft_prompt = soft_prompt.expand(-1, batch_size, -1)
        if hasattr(self, "soft_prompt_proj") and self.soft_prompt_proj is not None:
            soft_prompt = self.soft_prompt_proj(soft_prompt)
        return soft_prompt

    def _build_post_text_soft_prompt(
        self,
        batch_size: int,
        target_len: int,
        target_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        soft_prompt_param = getattr(self, "soft_prompt", None)
        if soft_prompt_param is None:
            return None

        soft_prompt = soft_prompt_param
        if soft_prompt.shape[1] != batch_size:
            soft_prompt = soft_prompt.expand(-1, batch_size, -1)
        if hasattr(self, "soft_prompt_proj") and self.soft_prompt_proj is not None:
            soft_prompt = self.soft_prompt_proj(soft_prompt)

        if soft_prompt.shape[2] != target_dim:
            raise RuntimeError(
                f"Post-text soft prompt dim mismatch: got {soft_prompt.shape[2]}, expected {target_dim}"
            )

        tuned_soft = torch.zeros(
            (target_len, batch_size, target_dim),
            device=device,
            dtype=dtype,
        )
        copy_len = min(int(soft_prompt.shape[0]), int(target_len))
        if copy_len > 0:
            tuned_soft[:copy_len] = soft_prompt[:copy_len].to(device=device, dtype=dtype)
        return tuned_soft

    def _apply_post_text_soft_prompt(self, text_outputs: Dict[str, torch.Tensor]):
        language_features = text_outputs.get("language_features", None)
        if language_features is None:
            raise KeyError("language_features missing in text outputs for post_text_add soft prompt.")

        if language_features.ndim != 3:
            raise RuntimeError(
                f"language_features should be [L, B, C], got {tuple(language_features.shape)}"
            )

        soft_prompt = self._build_post_text_soft_prompt(
            batch_size=int(language_features.shape[1]),
            target_len=int(language_features.shape[0]),
            target_dim=int(language_features.shape[2]),
            device=language_features.device,
            dtype=language_features.dtype,
        )
        if soft_prompt is None:
            return text_outputs

        if self.prompt_only:
            text_outputs["language_features"] = soft_prompt
        else:
            text_outputs["language_features"] = language_features + soft_prompt
        return text_outputs

    def _print_trainable_summary_once(self):
        if hasattr(self, "_printed_trainable_summary") and self._printed_trainable_summary:
            return

        total = 0
        trainable = 0
        names = []
        for n, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                names.append(n)

        print(
            f"[RSPromptSemanticModel] total_params={total}, "
            f"trainable_params={trainable}, frozen_params={total - trainable}"
        )
        for n in names[:50]:
            print(f"[RSPromptSemanticModel][TRAINABLE] {n}")
        if len(names) > 50:
            print(f"[RSPromptSemanticModel] ... and {len(names) - 50} more trainable params")

        self._printed_trainable_summary = True

    def _has_trainable_seg_path_params(self) -> bool:
        seg_prefixes = (
            "sam3.transformer.decoder.",
            "sam3.segmentation_head.",
        )
        for n, p in self.named_parameters():
            if any(n.startswith(pr) for pr in seg_prefixes) and p.requires_grad:
                return True
        return False

    @staticmethod
    def _anchor_requires_grad(out: Dict[str, torch.Tensor]) -> bool:
        sem = out.get("semantic_seg", None)
        ins = out.get("pred_masks", None)
        sem_grad = bool(torch.is_tensor(sem) and sem.requires_grad)
        ins_grad = bool(torch.is_tensor(ins) and ins.requires_grad)
        return sem_grad or ins_grad

    def _rerun_decoder_and_heads_without_act_ckpt(
        self,
        out: Dict[str, torch.Tensor],
        hs,
        backbone_out,
        encoder_out,
        find_input,
    ) -> Dict[str, torch.Tensor]:
        decoder = getattr(self.sam3, "transformer", None)
        decoder = getattr(decoder, "decoder", None)
        seg_head = getattr(self.sam3, "segmentation_head", None)

        prev_decoder_ckpt = getattr(decoder, "use_act_checkpoint", None)
        prev_seg_ckpt = getattr(self.sam3, "use_act_checkpoint_seg_head", None)
        prev_seg_head_ckpt = getattr(seg_head, "act_ckpt", None)

        try:
            if prev_decoder_ckpt is not None:
                decoder.use_act_checkpoint = False
            if prev_seg_ckpt is not None:
                self.sam3.use_act_checkpoint_seg_head = False
            if prev_seg_head_ckpt is not None:
                seg_head.act_ckpt = False

            grad_ctx = (
                torch.set_grad_enabled(True)
                if self.training and not torch.is_grad_enabled()
                else contextlib.nullcontext()
            )
            with grad_ctx:
                out = {}
                out, hs = self.sam3._run_decoder(
                    pos_embed=encoder_out["pos_embed"],
                    memory=encoder_out["encoder_hidden_states"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=encoder_out["prompt_after_enc"],
                    prompt_mask=encoder_out["prompt_mask"],
                    encoder_out=encoder_out,
                )
                self.sam3._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=encoder_out["encoder_hidden_states"],
                    prompt=encoder_out["prompt_after_enc"],
                    prompt_mask=encoder_out["prompt_mask"],
                    hs=hs,
                )
            return out
        finally:
            if prev_decoder_ckpt is not None:
                decoder.use_act_checkpoint = prev_decoder_ckpt
            if prev_seg_ckpt is not None:
                self.sam3.use_act_checkpoint_seg_head = prev_seg_ckpt
            if prev_seg_head_ckpt is not None:
                seg_head.act_ckpt = prev_seg_head_ckpt

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        self._decide_debug_step()
        self._print_trainable_summary_once()

        # 1) batched prompt input
        find_input = self._get_find_input(batch)
        self._check_batched_prompt_input(find_input, batch)

        self._dbg("[RSPromptSemanticModel] text_ids:", find_input.text_ids)
        self._dbg("[RSPromptSemanticModel] img_ids:", find_input.img_ids)

        # 2) batched images + captions
        captions = self._get_captions(batch)
        images = self._get_images(batch)

        self._dbg("[RSPromptSemanticModel] batch_size:", len(captions))
        self._dbg(
            "[RSPromptSemanticModel] num_prompts:",
            int(find_input.text_ids.numel()) if torch.is_tensor(find_input.text_ids) else None,
        )
        self._dbg("[RSPromptSemanticModel] captions:", captions)
        self._dbg("[RSPromptSemanticModel] captions_table_size:", len(captions))
        self._dbg("[RSPromptSemanticModel] text_ids list:", find_input.text_ids.tolist())
        self._dbg(
            "[RSPromptSemanticModel] images shape:",
            None if images is None else tuple(images.shape),
        )

        # 3) official backbone path
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.sam3.backbone.forward_image(images))

        device = self.device
        pre_text_soft_prompt = None
        if self.soft_prompt_pos == "pre_text":
            pre_text_soft_prompt = self._build_pre_text_soft_prompt(batch_size=len(captions))
        text_outputs = self.sam3.backbone.forward_text(
            captions,
            device=device,
            soft_prompt_embed=pre_text_soft_prompt,
        )
        if self.soft_prompt_pos == "post_text_add":
            text_outputs = self._apply_post_text_soft_prompt(text_outputs)
        backbone_out.update(text_outputs)

        # 4) official geometric prompt path
        geometric_prompt = self._build_geometric_prompt(find_input)

        self._dbg(
            "[RSPromptSemanticModel] input_boxes shape:",
            None if getattr(find_input, "input_boxes", None) is None
            else tuple(find_input.input_boxes.shape),
        )
        self._dbg(
            "[RSPromptSemanticModel] input_boxes_mask shape:",
            None if getattr(find_input, "input_boxes_mask", None) is None
            else tuple(find_input.input_boxes_mask.shape),
        )
        self._dbg(
            "[RSPromptSemanticModel] input_boxes_label shape:",
            None if getattr(find_input, "input_boxes_label", None) is None
            else tuple(find_input.input_boxes_label.shape),
        )

        # 5) official prompt encoding
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=None,
            visual_prompt_mask=None,
            encode_text=True,
            prev_mask_pred=None,
        )

        self._dbg(
            "[RSPromptSemanticModel] prompt shape:",
            None if prompt is None else tuple(prompt.shape),
        )
        self._dbg(
            "[RSPromptSemanticModel] prompt_mask shape:",
            None if prompt_mask is None else tuple(prompt_mask.shape),
        )

        # 6) prompt tuning
        prompt, prompt_mask = self._apply_prompt_tuning(prompt, prompt_mask)

        self._dbg(
            "[RSPromptSemanticModel] tuned prompt shape:",
            None if prompt is None else tuple(prompt.shape),
        )
        self._dbg(
            "[RSPromptSemanticModel] tuned prompt_mask shape:",
            None if prompt_mask is None else tuple(prompt_mask.shape),
        )

        # 7) encoder
        backbone_out, encoder_out, _ = self.sam3._run_encoder(
            backbone_out=backbone_out,
            find_input=find_input,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_extra_kwargs=None,
        )

        out: Dict[str, Any] = {}

        # 8) decoder + 9) segmentation heads
        force_disable_act_ckpt = os.environ.get("SAM3_RS_FORCE_DISABLE_ACT_CKPT", "0") in {
            "1",
            "true",
            "True",
        }
        decoder = getattr(self.sam3, "transformer", None)
        decoder = getattr(decoder, "decoder", None)
        seg_head = getattr(self.sam3, "segmentation_head", None)
        prev_decoder_ckpt = getattr(decoder, "use_act_checkpoint", None)
        prev_seg_ckpt = getattr(self.sam3, "use_act_checkpoint_seg_head", None)
        prev_seg_head_ckpt = getattr(seg_head, "act_ckpt", None)

        try:
            if force_disable_act_ckpt:
                if prev_decoder_ckpt is not None:
                    decoder.use_act_checkpoint = False
                if prev_seg_ckpt is not None:
                    self.sam3.use_act_checkpoint_seg_head = False
                if prev_seg_head_ckpt is not None:
                    seg_head.act_ckpt = False

            grad_ctx = (
                torch.set_grad_enabled(True)
                if self.training and not torch.is_grad_enabled()
                else contextlib.nullcontext()
            )
            with grad_ctx:
                out, hs = self.sam3._run_decoder(
                    pos_embed=encoder_out["pos_embed"],
                    memory=encoder_out["encoder_hidden_states"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=encoder_out["prompt_after_enc"],
                    prompt_mask=encoder_out["prompt_mask"],
                    encoder_out=encoder_out,
                )

                self.sam3._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=encoder_out["encoder_hidden_states"],
                    prompt=encoder_out["prompt_after_enc"],
                    prompt_mask=encoder_out["prompt_mask"],
                    hs=hs,
                )
        finally:
            if force_disable_act_ckpt:
                if prev_decoder_ckpt is not None:
                    decoder.use_act_checkpoint = prev_decoder_ckpt
                if prev_seg_ckpt is not None:
                    self.sam3.use_act_checkpoint_seg_head = prev_seg_ckpt
                if prev_seg_head_ckpt is not None:
                    seg_head.act_ckpt = prev_seg_head_ckpt

        if "semantic_seg" not in out:
            if "pred_masks" in out:
                self._dbg(
                    "[RSPromptSemanticModel] semantic_seg missing; pred_masks exists. "
                    "This can be valid when semantic supervision is derived from instance masks."
                )
            else:
                raise KeyError(
                    "Prompt semantic forward expected at least one of ['semantic_seg', 'pred_masks'] "
                    f"in model output, but got keys={list(out.keys())}"
                )

        if "semantic_seg" in out:
            self._dbg(
                "[RSPromptSemanticModel] semantic_seg shape:",
                tuple(out["semantic_seg"].shape),
            )
            self._dbg(
                "[RSPromptSemanticModel] semantic_seg requires_grad:",
                out["semantic_seg"].requires_grad,
            )

        semantic_anchor = out.get("semantic_seg", None)
        instance_anchor = out.get("pred_masks", None)
        grad_anchor = None
        if semantic_anchor is not None and semantic_anchor.requires_grad:
            grad_anchor = semantic_anchor
        elif instance_anchor is not None and instance_anchor.requires_grad:
            grad_anchor = instance_anchor
        else:
            grad_anchor = semantic_anchor if semantic_anchor is not None else instance_anchor

        if self.training and grad_anchor is not None and not grad_anchor.requires_grad:
            if self._has_trainable_seg_path_params():
                # Fallback: some activation-checkpoint paths under DDP+AMP can
                # break grad connectivity in this branch. Re-run decoder+heads
                # once with activation checkpointing disabled.
                out = self._rerun_decoder_and_heads_without_act_ckpt(
                    out=out,
                    hs=hs,
                    backbone_out=backbone_out,
                    encoder_out=encoder_out,
                    find_input=find_input,
                )
                if self._anchor_requires_grad(out):
                    return out

                sem2 = out.get("semantic_seg", None)
                ins2 = out.get("pred_masks", None)
                sem_grad = bool(torch.is_tensor(sem2) and sem2.requires_grad)
                ins_grad = bool(torch.is_tensor(ins2) and ins2.requires_grad)
                raise RuntimeError(
                    "Segmentation supervision anchor tensor does not require grad. "
                    "Trainable params exist on decoder/segmentation head path, "
                    "and fallback rerun without activation checkpointing still failed. "
                    f"semantic_seg.requires_grad={sem_grad}, pred_masks.requires_grad={ins_grad}"
                )
            self._dbg(
                "[RSPromptSemanticModel] segmentation anchor has no grad, "
                "but no trainable decoder/seg-head params are enabled; skip strict check."
            )

        return out

