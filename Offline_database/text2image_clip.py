import torch
import open_clip

# =====================
# 配置
# =====================
model_name = "ViT-B-32"
feature_path = "Offline_database/Million-AID_train_image_features_clip.pt"
K = 5
query_text = "water"

# 关键：本地权重路径（离线）
ckpt_path = "weights/clip/ViT-B-32.pt"

# =====================
# 加载 CLIP（官方权重）
# =====================
model, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained=ckpt_path
)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()

# =====================
# 加载 CLIP 图像特征
# =====================
data = torch.load(feature_path)
image_features = data["image_features"].cuda().float()  # [N, D]
image_paths = data["image_paths"]

# =====================
# 编码文本
# =====================
with torch.no_grad():
    text_tokens = tokenizer([query_text]).cuda()
    text_feat = model.encode_text(text_tokens)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

# =====================
# 相似度 & Top-K
# =====================
similarity = (text_feat @ image_features.T).squeeze(0)
topk_scores, topk_indices = similarity.topk(K)

print("CLIP Top-K 检索结果：")
for score, idx in zip(topk_scores, topk_indices):
    print(f"{image_paths[idx]}  score={score.item():.4f}")
