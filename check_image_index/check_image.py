import numpy as np
from PIL import Image
import os

# 替换成你实际的图片路径
img_path = "train_weights/Potsdam_20260129_173139/val_epoch_0/2_13.png"

if os.path.exists(img_path):
    mask = Image.open(img_path)
    mask_np = np.array(mask)
    
    print(f"Shape: {mask_np.shape}")
    print(f"Unique Values (Class IDs): {np.unique(mask_np)}")
    
    # 简单的可视化：将 ID 放大 40 倍，保存为新图方便肉眼观察
    vis_mask = Image.fromarray((mask_np * 40).astype(np.uint8))
    vis_mask.save("check_vis.png")
    print("已保存可视化图片为 check_vis.png")
else:
    print("文件不存在")