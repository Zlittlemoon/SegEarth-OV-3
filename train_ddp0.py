import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

# 引入你的模型和数据集
from model.gcsam3_segmentor import SegEarthOV3Segmentation 
from dataset.dataset import RemoteSensingDataset
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

# ================= 配置 =================
CONFIG = {
    "lr": 1e-4,             
    "batch_size": 1,        
    "grad_accum_steps": 4,  
    "epochs": 20,
    "crop_size": 512,       
    "img_dir": "data/isaid_full_image/img_dir/train",
    "mask_dir": "data/isaid_full_image/ann_dir/train",
    "class_file": "configs/cls_iSAID.txt",
    "use_adapter": True,
    "clip_ckpt": "weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
    "feature_db": "Offline_database/Million-AID_train_image_features.pt",
    "dataset_name": "iSAID" 
}

def setup_distributed():
    """初始化分布式环境"""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup():
    dist.destroy_process_group()

def draw_loss_curve(loss_list, save_path="loss_curve.png"):
    epochs = range(1, len(loss_list) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, 'b-o', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    is_master = (local_rank == 0)

    if is_master:
        print(f"Initializing Model on {torch.cuda.device_count()} GPUs...")

    # 1. 初始化模型
    model = SegEarthOV3Segmentation(
        classname_path=CONFIG["class_file"],
        use_adapter=CONFIG["use_adapter"],
        clip_ckpt_path=CONFIG["clip_ckpt"],
        feature_db_path=CONFIG["feature_db"],
        device=device
    ).to(device)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 2. 配置可训练参数
    trainable_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "adapter" in name or "router" in name or "gate" in name:
            param.requires_grad = True
            trainable_params.append(param)
    
    if is_master:
        print(f"Trainable params count: {len(trainable_params)}")

    # 3. DDP 包装
    # find_unused_parameters=True 依然需要，防止无关参数报错
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(trainable_params, lr=CONFIG["lr"], weight_decay=0.01)
    
    # 4. 数据加载
    train_dataset = RemoteSensingDataset(
        CONFIG["img_dir"], 
        CONFIG["mask_dir"], 
        CONFIG["class_file"],
        crop_size=CONFIG["crop_size"], 
        is_train=True 
    )
    
    sampler = DistributedSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False,      
        sampler=sampler,    
        num_workers=4,
        pin_memory=True
    )
    
    model.module.processor.model.eval()
    if hasattr(model.module, 'adapter'):
        model.module.adapter.train()

    loss_history = []

    # ================= 训练循环 =================
    for epoch in range(CONFIG["epochs"]):
        sampler.set_epoch(epoch)
        
        if is_master:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        else:
            pbar = train_loader

        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            img_path = batch["img_path"][0]
            gt_mask = batch["gt_mask"]
            inputs = batch["image"].to(device)
            
            data_sample = SegDataSample()
            data_sample.set_metainfo({"img_path": img_path})
            data_sample.gt_sem_seg = PixelData(data=gt_mask)
            data_samples = [data_sample]
            
            # [核心修复] 使用 no_sync 上下文
            # 这会禁止 DDP 在 backward 时自动通信，允许你在内部多次 backward
            with model.no_sync():
                loss_dict = model(inputs, data_samples, mode='loss')
            
            loss_val = loss_dict["loss_seg"]
            epoch_loss += loss_val
            
            # 梯度更新逻辑
            if (step + 1) % CONFIG["grad_accum_steps"] == 0:
                
                # [核心修复] 手动同步梯度
                # 因为我们禁用了自动同步，现在必须手动把所有卡的梯度加起来并平均
                for param in trainable_params:
                    if param.grad is not None:
                        # 1. 求和 (All-Reduce Sum)
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        # 2. 归一化 (核心修复点！)
                        # 我们需要除以 (显卡数量 * 累积步数)
                        # 比如 4张卡 * 4步累积 = 16。意味着我们把16份梯度加起来了，必须除以16回去。
                        divisor = dist.get_world_size() * CONFIG["grad_accum_steps"]
                        param.grad.data /= divisor

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if is_master:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        
        # 记录与保存
        if is_master:
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1} Mean Loss: {avg_loss:.4f}")
            
            os.makedirs("weights", exist_ok=True)
            save_name = f"weights/{CONFIG['dataset_name']}_ddp_adapter_epoch_{epoch+1}.pth"
            torch.save(model.module.adapter.state_dict(), save_name)
            print(f"Saved weights to {save_name}")

    if is_master:
        print("Training finished. Drawing loss curve...")
        curve_name = f"weights/{CONFIG['dataset_name']}_ddp_train_loss_curve.png"
        draw_loss_curve(loss_history, save_path=curve_name)
    
    cleanup()

if __name__ == "__main__":
    main()