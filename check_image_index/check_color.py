import numpy as np
from PIL import Image
import os
from rich import print

# Potsdam 数据集的标准配色方案 (RGB)
POTSDAM_PALETTE = {
    0: (255, 255, 255), # Impervious surfaces (Road) - 白色
    1: (0, 0, 255),     # Building - 蓝色
    2: (0, 255, 255),   # Low vegetation - 青色
    3: (0, 255, 0),     # Tree - 绿色
    4: (255, 255, 0),   # Car - 黄色
    5: (255, 0, 0),     # Clutter/Background - 红色
    255: (0, 0, 0)      # Ignore - 黑色
}

def colorize_mask(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label_id, color in POTSDAM_PALETTE.items():
        # 找到所有等于当前 label_id 的像素，填上颜色
        color_mask[mask_np == label_id] = color
        
    return Image.fromarray(color_mask)

# 替换成你实际的文件路径
img_path = "train_weights/Potsdam_20260129_173139/val_epoch_0/2_13.png"

if os.path.exists(img_path):
    mask = Image.open(img_path)
    mask_np = np.array(mask)
    
    print(f"处理中: {img_path}")
    print(f"包含类别: {np.unique(mask_np)}")
    
    # 生成彩色图
    vis_img = colorize_mask(mask_np)
    
    # 保存
    save_name = "check_color_vis.png"
    vis_img.save(save_name)
    print(f"✅ 已保存彩色可视化结果为: {save_name}")
    print("快去打开看看，应该是蓝色(建筑)、青色(草地)和绿色(树)的组合！")
else:
    print("❌ 文件路径不对，请检查路径")