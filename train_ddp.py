import os
# [新增] 必须放在 import torch 之前，防止 cv2 多线程与 DDP 冲突导致死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    pass

import shutil
import errno
import tempfile
import multiprocessing.util as mp_util
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 设置 Matplotlib 后端
plt.switch_backend('agg')

# MM系列库
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.evaluation import IoUMetric
from dataset.dataset import RemoteSensingDataset

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 你的模型
from model.gcsam3_segmentor import SegEarthOV3Segmentation

# Use a local repo tmp dir (avoid NFS /system tmp)
_local_tmp = os.path.abspath("tmp_multiproc")
os.makedirs(_local_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _local_tmp)
tempfile.tempdir = _local_tmp

try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

# Patch mp_util
if hasattr(mp_util, "_remove_temp_dir") and callable(mp_util._remove_temp_dir):
    _orig_remove = mp_util._remove_temp_dir
    def _safe_remove_temp_dir(name, warn=False):
        try:
            _orig_remove(name)
        except OSError as e:
            if getattr(e, "errno", None) == errno.EBUSY:
                print(f"Warning: could not remove temp dir {name}: EBUSY")
            else:
                raise
    try:
        mp_util._remove_temp_dir = _safe_remove_temp_dir
    except Exception:
        pass

# ==========================================
# 工具函数
# ==========================================
def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def draw_and_save_curves(loss_step_list, loss_epoch_list, miou_epoch_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss - Step
    plt.figure(figsize=(10, 6))
    steps = [x[0] for x in loss_step_list]
    vals = [x[1] for x in loss_step_list]
    plt.plot(steps, vals, label='Training Loss', color='blue', alpha=0.6)
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Loss vs Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_step_curve.png'))
    plt.close()

    # Loss - Epoch
    plt.figure(figsize=(10, 6))
    epochs = [x[0] for x in loss_epoch_list]
    vals = [x[1] for x in loss_epoch_list]
    plt.plot(epochs, vals, marker='o', label='Avg Epoch Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_epoch_curve.png'))
    plt.close()

    # mIoU - Epoch
    if len(miou_epoch_list) > 0:
        plt.figure(figsize=(10, 6))
        epochs = [x[0] for x in miou_epoch_list]
        vals = [x[1] for x in miou_epoch_list]
        plt.plot(epochs, vals, marker='s', label='Validation mIoU', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU (%)')
        plt.title('mIoU vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'miou_epoch_curve.png'))
        plt.close()

# ==========================================
# 分布式设置
# ==========================================
def setup():
    # [关键修改] 延长超时时间到 30 分钟，防止验证集卡死导致的 Timeout
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", timeout=timeout)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

# ==========================================
# 验证逻辑封装 (Sanity Check 和 Loop 通用)
# ==========================================
def run_validation(model, cfg_path, exp_dir, epoch, rank):
    """
    统一的验证函数，处理了路径共享和配置注入
    """
    # 确保模型处于评估模式
    model.eval()

    # 1. 定义共享收集目录 (绝对路径)
    # 放在 exp_dir 下，确保所有 Rank 都能访问到同一个物理路径
    val_name = f"val_epoch_{epoch+1}"
    collect_dir = os.path.abspath(os.path.join(exp_dir, f"dist_collect_epoch_{epoch+1}"))
    
    if rank == 0:
        os.makedirs(collect_dir, exist_ok=True)
    
    # 必须等待 Rank 0 创建好目录，否则 Rank 1 会报 FileNotFoundError
    dist.barrier()

    # 2. 准备 Config
    test_cfg = Config.fromfile(cfg_path)
    test_cfg.work_dir = os.path.join(exp_dir, val_name)
    os.makedirs(test_cfg.work_dir, exist_ok=True)

    # 3. [核心] 强制注入 collect_dir
    # 使用 merge_from_dict 保持代码简洁，但要确保覆盖到位
    # 注意：这里假设 test_evaluator 是 IoUMetric (dict 类型)
    if isinstance(test_cfg.test_evaluator, list):
        for metric in test_cfg.test_evaluator:
            metric['collect_dir'] = collect_dir
    else:
        test_cfg.merge_from_dict({
            'test_evaluator': {'collect_dir': collect_dir}
        })

    # 4. 运行验证
    runner = Runner.from_cfg(test_cfg)
    runner.model = model
    runner.load_checkpoint = lambda: None # 禁用自动加载，直接用当前模型权重

    results = runner.test()
    
    return results

# ==========================================
# 训练主函数
# ==========================================
def train(cfg_path):
    rank, local_rank, world_size = setup()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # 加载 Config
    cfg = Config.fromfile(cfg_path)

    # 实验目录初始化 (仅 Rank 0)
    exp_dir = None
    writer = None
    
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = cfg.get('dataset_type', 'CustomDataset').replace('Dataset', '')
        if 'Potsdam' in cfg_path: dataset_name = 'Potsdam'
        exp_dir = os.path.join("train_weights", f"{dataset_name}_{timestamp}")
        
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "vis"), exist_ok=True)
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        shutil.copy(cfg_path, os.path.join(exp_dir, "config_backup.py"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f"🚀 [Start] Experiment Directory: {exp_dir}")

    # 广播 exp_dir
    if rank == 0:
        exp_dir_bytes = exp_dir.encode('utf-8')
        exp_dir_tensor = torch.tensor(list(exp_dir_bytes), dtype=torch.uint8, device=device)
        exp_dir_len = torch.tensor(len(exp_dir_bytes), dtype=torch.int64, device=device)
    else:
        exp_dir_tensor = None
        exp_dir_len = torch.tensor(0, dtype=torch.int64, device=device)

    dist.broadcast(exp_dir_len, src=0)

    if rank != 0:
        exp_dir_tensor = torch.zeros(exp_dir_len.item(), dtype=torch.uint8, device=device)
    dist.broadcast(exp_dir_tensor, src=0)

    exp_dir = exp_dir_tensor.cpu().numpy().tobytes().decode('utf-8')

    # Logger 设置
    logger = logging.getLogger(f"train_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(exp_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 模型初始化
    model_cfg = cfg.model
    model = SegEarthOV3Segmentation(
        classname_path=model_cfg.classname_path,
        device=device,
        prob_thd=model_cfg.get('prob_thd', 0.0),
        bg_idx=model_cfg.get('bg_idx', 0),
        slide_stride=model_cfg.get('slide_stride', 0),
        slide_crop=model_cfg.get('slide_crop', 0),
        confidence_threshold=model_cfg.get('confidence_threshold', 0),
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        use_adapter=True,
        clip_ckpt_path="weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt",
        feature_db_path="Offline_database/Million-AID_train_image_features.pt", 
    ).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # [优化] find_unused_parameters=False 提高速度
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) 

    # 数据加载
    data_root = cfg.data_root
    train_img_dir = os.path.join(data_root, 'img_dir/train')
    train_mask_dir = os.path.join(data_root, 'ann_dir/train')

    if rank == 0:
        logger.info(f"Training set loaded from: img={train_img_dir}, mask={train_mask_dir}")

    train_dataset = RemoteSensingDataset(
            img_dir=train_img_dir,
            mask_dir=train_mask_dir,
            class_file=cfg.model.classname_path,
            crop_size=cfg.model.slide_crop,
            is_train=True
        )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

    if rank == 0:
        trainable_list = [(n, p.shape) for n, p in model.module.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters count: {len(trainable_list)}")
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    accum_steps = 16 
    
    loss_step_data = []
    loss_epoch_data = []
    miou_epoch_data = []
    
    global_step = 0
    best_miou = 0.0
    epochs = 20

    # ==========================================
    # Sanity Check (Zero-Shot Validation)
    # ==========================================
    if rank == 0:
        logger.info("Running Sanity Check (Zero-Shot Validation)...")
        if hasattr(model.module.adapter, 'gates'):
            nn.init.constant_(model.module.adapter.gates, 0.0)

    # 调用统一的验证函数
    # 注意：这里传入 model.module，因为 runner 内部不需要 DDP 包装
    # 但如果是多卡验证，runner 内部会处理 dist
    results = run_validation(model.module, cfg_path, exp_dir, epoch=-1, rank=rank)

    if rank == 0:
        miou = results.get('val/mIoU', results.get('mIoU', None))
        if miou is not None:
            logger.info(f"Validation mIoU: {miou:.4f}")
            writer.add_scalar('Val/mIoU', miou, 0)
            miou_epoch_data.append((0, miou))
            
            # 保存初始模型
            latest_path = os.path.join(exp_dir, "checkpoints", "latest.pth")
            torch.save(model.module.adapter.state_dict(), latest_path)

    # 验证完后同步一次
    dist.barrier()
    
    # ==========================================
    # 训练循环
    # ==========================================
    for epoch in range(epochs):
        model.train()
        model.apply(freeze_bn) 
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank != 0))
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # [Debug 1] 检查数据归一化
            if step == 0 and rank == 0:
                imgs_debug = batch['image']
                masks_debug = batch['gt_mask']
                logger.info("\n" + "="*40)
                logger.info(f"[DEBUG DATA] Epoch {epoch+1} Start")
                logger.info(f"Image Range: min={imgs_debug.min().item():.3f}, max={imgs_debug.max().item():.3f}, mean={imgs_debug.mean().item():.3f}")
                logger.info("="*40 + "\n")

            images = batch['image'].to(device)
            gt_masks = batch['gt_mask'].to(device)

            data_samples = []
            for i in range(len(images)):
                ds = SegDataSample()
                ds.set_metainfo({'img_path': batch['img_path'][i]})
                ds.gt_sem_seg = PixelData(data=gt_masks[i].unsqueeze(0))
                data_samples.append(ds)

            loss_dict = model(images, data_samples, mode='loss')
            loss = loss_dict.get('loss_seg', None)
            if loss is None: loss = list(loss_dict.values())[0]

            loss_scaled = loss / accum_steps
            loss_val = loss.item()

            is_nan = False
            if np.isnan(loss_val) or np.isinf(loss_val):
                logger.warning(f"[Rank {rank}] Found NaN/Inf loss at step {step}! Skipping step.")
                is_nan = True

            is_last_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))

            if not is_nan:
                if not is_last_step:
                    with model.no_sync():
                        loss_scaled.backward()
                else:
                    loss_scaled.backward()

            if is_last_step:
                if not is_nan:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss_val
            global_step += 1

            if rank == 0:
                writer.add_scalar('Train/Loss_Step', loss_val, global_step)
                loss_step_data.append((global_step, loss_val))
                pbar.set_postfix({'loss': loss_val})

        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor)
        
        local_steps = len(train_loader) 
        total_steps = local_steps * world_size
        avg_loss = total_loss_tensor.item() / (total_steps + 1e-6)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
            writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch + 1)
            loss_epoch_data.append((epoch + 1, avg_loss))

            if hasattr(model.module.adapter, 'gates'):
                gate_vals = model.module.adapter.gates.detach().cpu().numpy()
                logger.info(f"[DEBUG MODEL] Epoch {epoch+1} Adapter Gates: {gate_vals}")

        scheduler.step()

        # ==========================================
        # Loop 验证循环 (使用统一函数)
        # ==========================================
        if (epoch + 1) % 1 == 0:
            if rank == 0: logger.info("Starting Validation...")
            
            # 调用统一的验证函数
            results = run_validation(model.module, cfg_path, exp_dir, epoch=epoch, rank=rank)

            if rank == 0:
                miou = results.get('val/mIoU', results.get('mIoU', None))
                if miou is not None:
                    logger.info(f"Validation mIoU: {miou:.4f}")
                    writer.add_scalar('Val/mIoU', miou, epoch + 1)
                    miou_epoch_data.append((epoch + 1, miou))

                    if miou > best_miou:
                        best_miou = miou
                        save_path = os.path.join(exp_dir, "checkpoints", "best_miou.pth")
                        torch.save(model.module.adapter.state_dict(), save_path)
                        logger.info(f"🔥 New Best mIoU! Saved to {save_path}")

                    # 更新最新的 checkpoint 和 可视化
                    latest_path = os.path.join(exp_dir, "checkpoints", "latest.pth")
                    torch.save(model.module.adapter.state_dict(), latest_path)
                    vis_dir = os.path.join(exp_dir, "vis")
                    draw_and_save_curves(loss_step_data, loss_epoch_data, miou_epoch_data, vis_dir)

        dist.barrier()

    if rank == 0:
        vis_dir = os.path.join(exp_dir, "vis")
        draw_and_save_curves(loss_step_data, loss_epoch_data, miou_epoch_data, vis_dir)
        writer.close()
    cleanup()

def main():
    cfg_path = "configs/cfg_potsdam.py"
    train(cfg_path)

if __name__ == "__main__":
    main()