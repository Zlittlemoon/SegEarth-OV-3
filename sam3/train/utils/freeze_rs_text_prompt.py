import os
import re
from typing import Iterable, List, Tuple

import torch.nn as nn


MINIMAL_KEEP_REGEX = [
    r"^backbone\.language_backbone\.",
    r"(?:^|\.)(text)(?:_|\.|$)",
    r"(?:^|\.)(prompt)(?:_|\.|$)",
]

EXTENDED_EXTRA_KEEP_REGEX = [
    r"(?:^|\.)(presence)(?:_|\.|$)",
    r"(?:^|\.)(semantic)(?:_|\.|$)",
    r"(?:^|\.)(sem_seg)(?:_|\.|$)",
    r"(?:^|\.)(mask_decoder)(?:_|\.|$)",
    r"(?:^|\.)(class_embed)(?:_|\.|$)",
    r"(?:^|\.)(bbox_embed)(?:_|\.|$)",
]

BLOCK_REGEX = [
    r"^backbone\.vision_backbone\.",
    r"(?:^|\.)(tracker)(?:_|\.|$)",
    r"(?:^|\.)(memory)(?:_|\.|$)",
    r"(?:^|\.)(video)(?:_|\.|$)",
]

SR_ADAPTER_KEEP_REGEX = [
    r"^sr_adapter\.",
]

SR_EXTRACTOR_BLOCK_REGEX = [
    r"^sr_extractor\.",
]


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def freeze_for_remote_sensing_text_prompt_semseg(
    model: nn.Module,
    mode: str = "minimal",
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    mode = mode.lower().strip()
    if mode not in {"minimal", "extended"}:
        raise ValueError(f"Unsupported freeze mode: {mode}")

    keep_regex = list(MINIMAL_KEEP_REGEX)
    if mode == "extended":
        keep_regex += EXTENDED_EXTRA_KEEP_REGEX
    
    if os.environ.get("SAM3_RS_USE_SR", "0") == "1":
        keep_regex += SR_ADAPTER_KEEP_REGEX
    
    block_regex = list(BLOCK_REGEX) + SR_EXTRACTOR_BLOCK_REGEX

    trainable_names: List[str] = []
    frozen_names: List[str] = []

    for _, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        keep = _match_any(name, keep_regex)
        block = _match_any(name, block_regex)

        if keep and not block:
            p.requires_grad = True
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    n_total = 0
    n_trainable = 0
    for p in model.parameters():
        n = p.numel()
        n_total += n
        if p.requires_grad:
            n_trainable += n

    print("=" * 100)
    print(f"[RS FREEZE] mode={mode}")
    print(f"[RS FREEZE] Total params     : {n_total:,}")
    print(f"[RS FREEZE] Trainable params : {n_trainable:,}")
    print(f"[RS FREEZE] Frozen params    : {n_total - n_trainable:,}")
    print(f"[RS FREEZE] Num trainable tensors: {len(trainable_names)}")
    print("[RS FREEZE] First 30 trainable names:")
    for n in trainable_names[:30]:
        print(f"  [T] {n}")
    print("=" * 100)

    return trainable_names, frozen_names


def maybe_apply_remote_sensing_freeze(model):
    flag = os.environ.get("SAM3_RS_FREEZE", "0")
    mode = os.environ.get("SAM3_RS_TRAIN_MODE", "minimal")

    print(f"[RS FREEZE] env SAM3_RS_FREEZE={flag}")
    print(f"[RS FREEZE] env SAM3_RS_TRAIN_MODE={mode}")

    if flag != "1":
        print("[RS FREEZE] skip freezing because SAM3_RS_FREEZE != 1")
        return model

    trainable_names, frozen_names = freeze_for_remote_sensing_text_prompt_semseg(
        model,
        mode=mode,
        verbose=True,
    )

    if len(trainable_names) == 0:
        raise RuntimeError(
            "[RS FREEZE] No trainable parameters left after freeze."
        )

    return model