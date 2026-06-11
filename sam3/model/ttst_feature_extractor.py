import torch
import torch.nn as nn
import torch.nn.functional as F

# 你需要把 TTST_arc.py 放到工程可 import 的位置
# 例如 third_party/TTST/model_archs/TTST_arc.py
from third_party.TTST.model_archs.TTST_arc import TTST


class TTSTFeatureExtractor(nn.Module):
    def __init__(
        self,
        ckpt_path,
        input_size=256,
        sr_dim=180,
        device="cuda",
    ):
        super().__init__()
        self.input_size = input_size

        self.net = TTST(
            img_size=input_size,
            upscale=4,
            embed_dim=sr_dim,
            upsampler="pixelshuffle",
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict):
            for key in ["params_ema", "params", "state_dict", "model", "net"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    print(f"[TTSTFeatureExtractor] using checkpoint key: {key}")
                    break
            else:
                state = ckpt
        else:
            state = ckpt


        def strip_prefix(state_dict, prefix):
            if not isinstance(state_dict, dict):
                return state_dict

            keys = list(state_dict.keys())
            if len(keys) == 0:
                return state_dict

            # 只在所有 key 都有该前缀时统一去掉，避免误删
            if all(k.startswith(prefix) for k in keys):
                print(f"[TTSTFeatureExtractor] strip prefix: {prefix}")
                return {k[len(prefix):]: v for k, v in state_dict.items()}

            return state_dict


        for prefix in ["module.", "net.", "model.", "generator.", "G."]:
            state = strip_prefix(state, prefix)

        model_keys = set(self.net.state_dict().keys())
        state_keys = set(state.keys())
        overlap = len(model_keys & state_keys)

        print(
            f"[TTSTFeatureExtractor] key overlap: "
            f"{overlap}/{len(model_keys)} model keys, ckpt keys={len(state_keys)}"
        )

        missing, unexpected = self.net.load_state_dict(state, strict=False)

        print(f"[TTSTFeatureExtractor] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[TTSTFeatureExtractor] first missing:", missing[:20])
        if len(unexpected) > 0:
            print("[TTSTFeatureExtractor] first unexpected:", unexpected[:20])

        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.to(device)

    @staticmethod
    def _to_01(x):
        # SAM3 输入通常是 [-1, 1]，TTST 通常使用 [0, 1]
        if x.min().item() < -0.1:
            x = x * 0.5 + 0.5
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, x):
        x = self._to_01(x)

        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        net = self.net
        feat = net.conv_first(x)
        body_feat = net.conv_after_body(net.forward_features(feat)) + feat
        return body_feat