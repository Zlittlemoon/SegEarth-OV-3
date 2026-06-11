import torch
import torch.nn as nn
import torch.nn.functional as F


class SRFeatureAdapter(nn.Module):
    def __init__(
        self,
        sr_dim=180,
        sam_dim=256,
        num_levels=3,
        residual_scale=0.1,
        init_scale=1e-3,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.residual_scale = residual_scale

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sr_dim, sam_dim, kernel_size=1),
                nn.GroupNorm(8, sam_dim),
                nn.GELU(),
            )
            for _ in range(num_levels)
        ])

        self.gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sam_dim * 2, sam_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(sam_dim, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            for _ in range(num_levels)
        ])

        self.level_scale = nn.Parameter(torch.ones(num_levels) * init_scale)

    def forward(self, fpn_feats, sr_feat):
        enhanced = []

        for i, feat in enumerate(fpn_feats):
            if i >= self.num_levels:
                enhanced.append(feat)
                continue

            sr_i = F.interpolate(
                sr_feat,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            sr_i = sr_i.to(device=feat.device, dtype=feat.dtype)
            sr_i = self.proj[i](sr_i)

            gate_i = self.gate[i](torch.cat([feat, sr_i], dim=1))
            scale_i = torch.tanh(self.level_scale[i]) * self.residual_scale

            enhanced.append(feat + scale_i * gate_i * sr_i)

        return enhanced