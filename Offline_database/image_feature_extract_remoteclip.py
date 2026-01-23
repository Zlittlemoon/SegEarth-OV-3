import torch
import open_clip
from PIL import Image
import os
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# =====================
# 配置
# =====================
model_name = "ViT-B-32"
ckpt_path = "weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
image_dir = "data/Million-AID"
save_path = "Offline_database/Million-AID_train_image_features.pt"
batch_size = 32

# =====================
# 加载模型
# =====================
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
message = model.load_state_dict(ckpt)
print(message)

model = model.cuda().eval()

# =====================
# 读取图片路径
# =====================
image_paths = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images.")

all_features = []
valid_paths = []

# =====================
# 批量提取特征
# =====================
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
                valid_paths.append(p)
            except:
                print(f"Skip {p}")

        if len(images) == 0:
            continue

        images = torch.stack(images).cuda()
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu())

# =====================
# 保存特征
# =====================
image_features = torch.cat(all_features, dim=0)

torch.save(
    {
        "image_features": image_features,  # [N, D]
        "image_paths": valid_paths
    },
    save_path
)

print(f"Saved {image_features.shape[0]} image features to {save_path}")
