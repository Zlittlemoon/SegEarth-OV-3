import torch
import clip
from PIL import Image
import os
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

device = "cuda"
model, preprocess = clip.load(
    "ViT-B/32",
    device=device,
    jit=False,
    download_root="weights/clip"
)

model.eval()

image_dir = "data/Million-AID"
save_path = "Offline_database/Million-AID_train_image_features_clip.pt"

image_paths = []
for root, _, files in os.walk(image_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, f))

all_features = []
valid_paths = []

with torch.no_grad():
    for path in tqdm(image_paths):
        try:
            image = preprocess(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(device)

            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)

            all_features.append(feat.cpu())
            valid_paths.append(path)
        except Exception as e:
            print(f"Skip {path}: {e}")

image_features = torch.cat(all_features, dim=0)

torch.save(
    {
        "image_features": image_features,  # [N, 512]
        "image_paths": valid_paths
    },
    save_path
)

print(f"Saved {image_features.shape[0]} features to {save_path}")
