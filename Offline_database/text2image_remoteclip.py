import torch
import open_clip

# =====================
# 配置
# =====================
model_name = "ViT-B-32"
ckpt_path = "weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
feature_path = "Offline_database/Million-AID_train_image_features.pt"
K = 5

query_text = "water"

# =====================
# 加载模型
# =====================
model, _, _ = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
message = model.load_state_dict(ckpt)
print(message)

model = model.cuda().eval()

# =====================
# 加载图像特征
# =====================
data = torch.load(feature_path)
image_features = data["image_features"].cuda()  # [N, D]
image_paths = data["image_paths"]

# =====================
# 编码文本
# =====================
with torch.no_grad():
    text_tokens = tokenizer([query_text]).cuda()
    text_feat = model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# =====================
# 相似度 & Top-K
# =====================
similarity = (text_feat @ image_features.T).squeeze(0)
topk_scores, topk_indices = similarity.topk(K)

print("Top-K 检索结果：")
for score, idx in zip(topk_scores, topk_indices):
    print(f"{image_paths[idx]}  score={score.item():.4f}")
