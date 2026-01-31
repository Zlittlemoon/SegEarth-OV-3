import torch
import torch.nn as nn
import numpy as np

class RemoteSensingAdapter(nn.Module):
    def __init__(self, clip_dim=512, sam_dim=256, num_heads=8, num_levels=3, topk=5, residual_scale=0.01):
        super().__init__()
        self.proj_proto = nn.Linear(clip_dim, sam_dim)
        self.cross_attn = nn.MultiheadAttention(sam_dim, num_heads, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # [之前讨论的修复] 显式初始化为 0
        self.gates = nn.Parameter(torch.zeros(num_levels)) 
        
        self.residual_scale = residual_scale 
        self.norm = nn.LayerNorm(sam_dim)
        self.topk = topk

    def forward(self, fpn_feats: list, prototypes: torch.Tensor, proto_scores: torch.Tensor = None):
        """
        fpn_feats: List[(B, C, H_i, W_i)]
        prototypes: [B, topk, clip_dim] 或 [topk, clip_dim]
        proto_scores: [B, topk] 或 [topk]
        """
        # [FIX] 如果所有 gates 都接近 0（未训练状态），直接返回原始特征
        # 这确保 epoch 0 时 adapter 完全透明，与 baseline 行为一致
        if torch.allclose(self.gates, torch.zeros_like(self.gates), atol=1e-6):
            return fpn_feats

        B = fpn_feats[0].shape[0]
        num_levels = len(fpn_feats)

        # 自动广播 batch 维度
        if prototypes.dim() == 2:
            prototypes = prototypes.unsqueeze(0).expand(B, -1, -1)

        if proto_scores is not None:
            # 自动广播
            if proto_scores.dim() == 1:
                proto_scores = proto_scores.unsqueeze(0).expand(B, -1)

            # 【核心修复】强制将 proto_scores 移动到 GPU (和 prototypes 一致)
            # 这一行解决了 "found at least two devices, cuda:x and cpu!" 错误
            proto_scores = proto_scores.to(prototypes.device)

        # 提取 global
        global_feat = self.global_pool(fpn_feats[-1]).flatten(2).transpose(1, 2).detach()

        proto_proj = self.proj_proto(prototypes)

        if proto_scores is not None:
            # 现在都在同一个设备上了，可以相乘
            proto_proj = proto_proj * proto_scores.unsqueeze(-1)

        # memory: 仅阻断 global_feat 的梯度
        memory = torch.cat([proto_proj, global_feat], dim=1)

        enhanced_feats = []
        chunk_size = 8192

        for level_idx in range(num_levels):
            feat = fpn_feats[level_idx]
            q_local = feat.flatten(2).transpose(1, 2)

            # 分块 Attention 逻辑
            if level_idx == 0 and q_local.shape[1] > chunk_size:
                outs = []
                for s in range(0, q_local.shape[1], chunk_size):
                    e = min(s + chunk_size, q_local.shape[1])
                    chunk_q = q_local[:, s:e, :]
                    chunk_out, _ = self.cross_attn(chunk_q, memory, memory)
                    outs.append(chunk_out)
                attn_out = torch.cat(outs, dim=1)
            else:
                attn_out, _ = self.cross_attn(q_local, memory, memory)

            # [之前讨论的修复] 去掉 sigmoid 限制，直接学习门控
            gate = self.gates[level_idx]
            enhanced = q_local + gate * attn_out # 去掉了 .unsqueeze(0).unsqueeze(0)，因为 gate 是标量广播即可

            enhanced = self.norm(enhanced)
            enhanced_feats.append(enhanced.transpose(1, 2).view_as(feat))

        return enhanced_feats