import torch
import torch.nn as nn

class DualSourceAdapter(nn.Module):
    def __init__(self, clip_dim=512, sam_dim=256):
        """
        clip_dim: RemoteCLIP输出维度 (ViT-B-32是512)
        sam_dim: SAM MaskDecoder输入维度 (通常是256)
        """
        super().__init__()
        # 1. 维度对齐层：把CLIP特征映射到SAM空间
        self.proj = nn.Linear(clip_dim, sam_dim)
        
        # 2. 交叉注意力层
        # query=SAM特征, key/value=混合Memory(原型+全局)
        self.cross_attn = nn.MultiheadAttention(embed_dim=sam_dim, num_heads=8, batch_first=True)
        
        # 3. 门控参数 & Norm
        self.norm = nn.LayerNorm(sam_dim)
        self.gate = nn.Parameter(torch.tensor(0.0)) # 初始为0，保证冷启动稳定

    def forward(self, q_local, v_ref, v_scores, v_global=None):
        """
        q_local:  [B, N_local, 256] (也就是 sam_feat, 必须是展平后的)
        v_ref:    [B, K, 512]       (检索出的Top-K特征)
        v_scores: [B, K]            (检索分数)
        v_global: [B, N_global, 256] (全局上下文，通常是q_local本身或者下采样的特征)
        """
        # --- 1. 维度投影 ---
        # [B, K, 512] -> [B, K, 256]
        ref_proj = self.proj(v_ref)
        
        # --- 2. 分数加权 (Soft Reweighting) ---
        if v_scores is not None:
            # v_scores: [B, K] -> [B, K, 1]
            # 这一步让低分检索图变暗，高分变亮
            ref_proj = ref_proj * v_scores.unsqueeze(-1)
            
        # --- 3. 构建 Memory (Hybrid Knowledge) ---
        # 核心修改：将外部知识(ref)与内部知识(global)结合
        if v_global is not None:
            # 在序列维度拼接: [B, K + N_global, 256]
            memory = torch.cat([ref_proj, v_global], dim=1)
        else:
            # 保底：如果没有global，只用检索特征
            memory = ref_proj 
        
        # --- 4. Cross Attention ---
        # query=q_local, key=memory, value=memory
        attn_out, _ = self.cross_attn(query=q_local, key=memory, value=memory)
        
        # --- 5. Residual Injection (残差注入) ---
        out = q_local + self.gate * attn_out
        
        return self.norm(out)