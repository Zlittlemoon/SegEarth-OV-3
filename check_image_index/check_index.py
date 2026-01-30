import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

def check_rs_dataset_labels(dataset_path, file_exts=['.tif', '.png', '.jpg']):
    """
    检查遥感数据集标签的值分布和属性。
    
    Args:
        dataset_path (str): 标签文件夹路径
        file_exts (list): 需要搜索的文件后缀列表
    """
    # 1. 获取所有文件路径
    image_paths = []
    for ext in file_exts:
        # 递归搜索 (包括子文件夹)
        image_paths.extend(list(Path(dataset_path).rglob(f"*{ext}")))
    
    if not image_paths:
        print(f"❌ 在路径 {dataset_path} 下未找到指定格式的文件。")
        return

    print(f"📂 找到 {len(image_paths)} 个标签文件，开始分析...")

    # 全局统计变量
    global_classes = set()
    global_pixel_counts = {}
    error_files = []
    
    # 2. 遍历文件
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            # 读取图片，flag=-1 (IMREAD_UNCHANGED) 保证读取原始深度和通道
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                error_files.append((str(img_path), "无法读取"))
                continue

            # 检查维度：遥感标签通常应该是二维数组 (H, W)
            if img.ndim == 3:
                # 某些情况下标签可能是 (H, W, 1) 或者错误的 (H, W, 3)
                if img.shape[2] == 1:
                    img = img.squeeze() # 压缩成 2D
                else:
                    # 如果是RGB彩色标签，通常需要映射表，这里仅记录unique值
                    # 提示：如果是RGB标签，建议转为灰度或索引图
                    pass 

            # 获取当前图片的唯一值和计数
            unique_values, counts = np.unique(img, return_counts=True)
            
            # 更新全局统计
            for val, count in zip(unique_values, counts):
                val = int(val) # 转换为原生 Python int
                global_classes.add(val)
                if val in global_pixel_counts:
                    global_pixel_counts[val] += count
                else:
                    global_pixel_counts[val] = count
                    
        except Exception as e:
            error_files.append((str(img_path), str(e)))

    # 3. 输出报告
    print("\n" + "="*40)
    print("📊 数据集标签分析报告")
    print("="*40)
    
    sorted_classes = sorted(list(global_classes))
    print(f"✅ 包含的类别值 (Class IDs): {sorted_classes}")
    
    total_pixels = sum(global_pixel_counts.values())
    print(f"\n📈 像素分布统计 (总像素: {total_pixels}):")
    print(f"{'Class ID':<10} {'Pixel Count':<15} {'Percentage':<10}")
    print("-" * 35)
    
    for cls_id in sorted_classes:
        count = global_pixel_counts[cls_id]
        ratio = (count / total_pixels) * 100
        print(f"{cls_id:<10} {count:<15} {ratio:.4f}%")
        
    # 4. 常见问题提示
    print("-" * 35)
    if 255 in sorted_classes:
        print("⚠️  注意: 检测到值 255。这通常是'忽略区域'(Ignore Index)或NoData。")
    
    if len(sorted_classes) > 20:
        print(f"⚠️  注意: 检测到 {len(sorted_classes)} 个类别，这对于语义分割来说可能偏多，请确认是否包含噪声值。")

    if error_files:
        print(f"\n❌ 处理失败的文件 ({len(error_files)}个):")
        for f, err in error_files[:5]: # 只显示前5个
            print(f"  - {f}: {err}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 把这里的路径换成你数据集标签所在的文件夹路径
    label_folder_path = r"data/potsdam_full_image/ann_dir/val" 
    
    check_rs_dataset_labels(label_folder_path)