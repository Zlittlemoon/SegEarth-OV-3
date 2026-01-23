import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from torchvision import transforms

class RemoteSensingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, class_file, crop_size=512, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.is_train = is_train  # 确保这里被正确赋值
        
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        with open(class_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_valid_classes = len(self.class_names)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) 
        
        w, h = image.size
        
        # ==========================================
        # 裁剪逻辑：只有 is_train=True 才会执行
        # ==========================================
        if self.is_train:
            # 1. 如果图片小于裁剪尺寸，进行填充
            if w < self.crop_size or h < self.crop_size:
                pad_w = max(self.crop_size - w, 0)
                pad_h = max(self.crop_size - h, 0)
                image = transforms.Pad((0, 0, pad_w, pad_h))(image)
                mask = transforms.Pad((0, 0, pad_w, pad_h), fill=255)(mask)
                w, h = image.size # 更新尺寸
            
            # 2. 随机裁剪
            i = random.randint(0, h - self.crop_size)
            j = random.randint(0, w - self.crop_size)
            
            image = image.crop((j, i, j + self.crop_size, i + self.crop_size))
            mask = mask.crop((j, i, j + self.crop_size, i + self.crop_size))
            
            # [调试用] 确保裁剪后尺寸正确
            # print(f"Cropped to: {image.size}") 
        
        image_tensor = transforms.ToTensor()(image)
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        # 标签清洗 (防止 index out of bounds)
        mask_tensor[mask_tensor >= self.num_valid_classes] = 255
        
        return {
            "image": image_tensor, 
            "gt_mask": mask_tensor,
            "img_path": img_path,
            "classes": self.class_names
        }