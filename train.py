import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import torch.nn.functional as F

# 引入你的模型和数据集
from model.gcsam3_segmentor import SegEarthOV3Segmentation 
from dataset.dataset import RemoteSensingDataset

# ================= 配置 =================
CONFIG = {
    "lr": 1e-4,             
    "batch_size": 1,        # 建议设为 1，因为 _forward 里是单张处理的
    "grad_accum_steps": 4,  # [新增] 梯度累积步数，模拟 Batch Size = 4
    "epochs": 20,
    "crop_size": 512,       # Dataset 依然保持 512 裁剪
    "device": "cuda",
    "img_dir": "data/isaid_full_image/img_dir/train",
    "mask_dir": "data/isaid_full_image/ann_dir/train",
    "class_file": "configs/cls_iSAID.txt",
    # Adapter 配置
    "use_adapter": True,
    "clip_ckpt": "weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
    "feature_db": "Offline_database/Million-AID_train_image_features.pt"
}

def draw_loss_curve(loss_list, save_path="loss_curve.png"):
    """
    绘制训练 Loss 曲线
    """
    epochs = range(1, len(loss_list) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, 'b-o', label='Training Loss')
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(save_path)
    plt.close() # 关闭画布，释放内存
    print(f"Loss curve saved to {save_path}")

def main():
    device = torch.device(CONFIG["device"])
    
    # 1. 初始化模型
    print("Initializing Model...")
    model = SegEarthOV3Segmentation(
        classname_path=CONFIG["class_file"],
        use_adapter=CONFIG["use_adapter"],
        clip_ckpt_path=CONFIG["clip_ckpt"],
        feature_db_path=CONFIG["feature_db"],
        device=device
    ).to(device)
    
    # ====================================================
    # [核心] 冻结参数策略 (Freeze Strategy)
    # ====================================================
    print("Configuring Trainable Parameters...")
    trainable_params = []
    
    for name, param in model.named_parameters():
        # 默认全部冻结
        param.requires_grad = False
        
        # 只解冻 Adapter 和 相关的 Head
        if "adapter" in name or "router" in name or "gate" in name:
            param.requires_grad = True
            trainable_params.append(param)
            print(f" -> Unlock: {name}")
            
    print(f"Trainable params count: {len(trainable_params)}")
    
    # 2. 优化器
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["lr"], weight_decay=0.01)
    
    # 3. 损失函数 (Standard PyTorch CE Loss)
    # criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 4. 数据加载
    train_dataset = RemoteSensingDataset(
        CONFIG["img_dir"], 
        CONFIG["mask_dir"], 
        CONFIG["class_file"],
        crop_size=512,  # [新增] 设定裁剪大小，SAM通常建议1024
        is_train=True    # [新增] 开启裁剪
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    
    # ================= 训练循环 =================
    model.train() 
    # 强制 SAM 内部冻结模块保持 eval
    model.processor.model.eval() 
    if hasattr(model, 'adapter'):
        model.adapter.train()

    # [新增] 用于记录每个 Epoch 的平均 Loss
    loss_history = []

    # [修改] 训练循环
    for epoch in range(CONFIG["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0
        
        optimizer.zero_grad() # 在循环外初始化清零
        
        for step, batch in enumerate(pbar):
            # 1. 拿到数据
            # 注意：现在 dataset 出来的需要构造成 mmseg 格式的 data_samples
            # 或者你直接把 loss 函数里的接口改一下，上面的 loss 代码是基于 mmseg 接口写的
            # 为了适配你目前的 train.py，我们需要封装一下 inputs 和 data_samples
            
            # --- 适配层 ---
            from mmseg.structures import SegDataSample
            from mmengine.structures import PixelData
            
            img_path = batch["img_path"][0] # 假设 dataset 返回了这个
            gt_mask = batch["gt_mask"]      # [1, H, W]
            inputs = batch["image"].to(device)
            
            data_sample = SegDataSample()
            data_sample.set_metainfo({"img_path": img_path})
            data_sample.gt_sem_seg = PixelData(data=gt_mask)
            data_samples = [data_sample]
            # ----------------

            # 2. 调用模型 (这会自动调用 loss 方法)
            # 注意：这里我们不调用 forward，而是直接调用 model.loss 
            # (因为 SegEarthOV3Segmentation 是 BaseSegmentor，可以直接 model(..., mode='loss'))
            # 但你手动写了 train loop，直接调 loss 方法最稳
            
            loss_dict = model.loss(inputs, data_samples)
            
            # 3. 获取 Loss 数值用于打印
            loss_val = loss_dict["loss_seg"]

            # [新增] 双重保险：防止 NaN 污染 epoch_loss
            if not math.isnan(loss_val):
                epoch_loss += loss_val
            else:
                if torch.distributed.get_rank() == 0:
                    print(f"Warning: NaN loss detected at step {step}")

            # [重要修复] 累加 Loss
            epoch_loss += loss_val  # <--- 漏了这一行！
            
            # 4. 更新权重
            # 注意：不需要 loss.backward() 了，因为在 model.loss 里已经做过了！
            
            if (step + 1) % CONFIG["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Mean Loss: {avg_loss:.4f}")
        
        # 保存权重
        os.makedirs("weights", exist_ok=True)
        torch.save(model.adapter.state_dict(), f"weights/adapter_epoch_{epoch+1}.pth")

        # ================= 训练结束，绘制图像 =================
    print("Training finished. Drawing loss curve...")
    draw_loss_curve(loss_history, save_path="weights/train_loss_curve.png")

if __name__ == "__main__":
    main()