from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = 'resources/loveda_2524.png'  # <-- 和你 demo 中一样的路径
gt_path = 'data/LoveDA/ann_dir/val/2524.png'                  # <-- 你的 GT 路径

# 读取原图
img = Image.open(img_path)  # <-- 新增这行

# 读取 GT
gt_img = Image.open(gt_path)
if gt_img.mode == 'P':
    gt_array = np.array(gt_img)
    print(f"Unique labels in GT: {np.unique(gt_array)}")
else:
    gt_array = np.array(gt_img)
    if gt_array.ndim == 3:
        gt_array = gt_array[:, :, 0]

# 并排显示：原图 + GT
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(gt_array, cmap='viridis')
ax[1].set_title('Ground Truth (viridis)')
ax[1].axis('off')

plt.tight_layout()
plt.savefig('loveda_2524_original_and_gt_viridis.png', bbox_inches='tight', dpi=300)
plt.show()
print("GT unique values:", np.unique(gt_array))