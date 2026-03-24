from __future__ import annotations

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
        self.prompt_only = os.environ.get("SAM3_RS_PROMPT_ONLY", "1") == "1"

        if self.prompt_tuning_mode == "soft":
            self.soft_prompt = nn.Parameter(
                torch.zeros(self.soft_prompt_len, 1, self.prompt_dim)
            )
            nn.init.normal_(self.soft_prompt, std=0.02)

        elif self.prompt_tuning_mode == "mlp":
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

    def _apply_prompt_tuning(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        prompt: [L, B, C]
        """
        if self.prompt_tuning_mode == "soft":
            if prompt.shape[-1] != self.prompt_dim:
                raise RuntimeError(
                    f"Prompt dim mismatch: prompt.shape={tuple(prompt.shape)}, "
                    f"expected last dim={self.prompt_dim}"
                )
            if prompt.shape[0] != self.soft_prompt.shape[0]:
                raise RuntimeError(
                    f"Soft prompt length mismatch: prompt.shape={tuple(prompt.shape)}, "
                    f"soft_prompt.shape={tuple(self.soft_prompt.shape)}. "
                    f"Set SAM3_RS_SOFT_PROMPT_LEN to match actual prompt length."
                )
            # soft_prompt: [L,1,C], prompt: [L,B,C] -> broadcast on B
            return prompt + self.soft_prompt

        if self.prompt_tuning_mode == "mlp":
            if prompt.shape[-1] != self.prompt_dim:
                raise RuntimeError(
                    f"Prompt dim mismatch: prompt.shape={tuple(prompt.shape)}, "
                    f"expected last dim={self.prompt_dim}"
                )
            return prompt + self.prompt_mlp(prompt)

        return prompt

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
        text_outputs = self.sam3.backbone.forward_text(captions, device=device)
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
        prompt = self._apply_prompt_tuning(prompt)

        self._dbg(
            "[RSPromptSemanticModel] tuned prompt shape:",
            None if prompt is None else tuple(prompt.shape),
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

        # 8) decoder
        out, hs = self.sam3._run_decoder(
            pos_embed=encoder_out["pos_embed"],
            memory=encoder_out["encoder_hidden_states"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=encoder_out["prompt_after_enc"],
            prompt_mask=encoder_out["prompt_mask"],
            encoder_out=encoder_out,
        )

        # 9) segmentation heads
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

        if "semantic_seg" not in out:
            raise KeyError(
                f"Prompt semantic forward expected 'semantic_seg' in model output, "
                f"but got keys={list(out.keys())}"
            )

        self._dbg(
            "[RSPromptSemanticModel] semantic_seg shape:",
            tuple(out["semantic_seg"].shape),
        )
        self._dbg(
            "[RSPromptSemanticModel] semantic_seg requires_grad:",
            out["semantic_seg"].requires_grad,
        )

        if self.training and not out["semantic_seg"].requires_grad:
            raise RuntimeError(
                "semantic_seg does not require grad. "
                "Most likely all parameters on the semantic segmentation path are frozen "
                "or prompt tuning parameters were not registered correctly."
            )

        return out
