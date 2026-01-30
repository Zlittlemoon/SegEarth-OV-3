import os
import os.path as osp
import argparse
import openpyxl
import torch
import torch.distributed as dist
from mmengine.dist import init_dist
from mmengine.runner import Runner
from mmengine.config import Config, DictAction
from datetime import timedelta

# 引入必要的模块
try:
    from model.gcsam3_segmentor import SegEarthOV3Segmentation
    import custom_datasets
except ImportError:
    pass 

def parse_args():
    parser = argparse.ArgumentParser(description='CorrCLIP evaluation with MMSeg')
    parser.add_argument('config', default='./configs/cfg_loveda.py')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show_dir', default='./show_dir/', help='directory to save visualizaion images')
    parser.add_argument('--out', type=str, help='The directory to save output prediction for offline evaluation')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def append_experiment_result(file_path, experiment_data):
    try:
        workbook = openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
    sheet = workbook.active
    if sheet['A1'].value is None:
        sheet.append(['Model', 'Dataset', 'aAcc', 'mIoU', 'mAcc'])
    for result in experiment_data:
        sheet.append([
            result.get('Model', 'N/A'),
            result.get('Dataset', 'N/A'),
            result.get('aAcc', 0),
            result.get('mIoU', 0),
            result.get('mAcc', 0)
        ])
    workbook.save(file_path)

def main():
    args = parse_args()
    
    # 1. 分布式初始化
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200))
            print(f"[Init] Rank {local_rank} initialized & bound to GPU {local_rank}")

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 2. Config 修复
    cfg.env_cfg = dict(
        cudnn_benchmark=True,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl', timeout=7200) 
    )

    trained_adapter_path = "weights/Potsdam_ddp_adapter_epoch_20.pth"
    if os.path.exists(trained_adapter_path):
        cfg.model.adapter_ckpt_path = trained_adapter_path
    
    runner = Runner.from_cfg(cfg)
    torch.cuda.empty_cache()

    # ==========================================================
    # [👑 终极修复]：强制开启 Flash Attention (SDPA) + FP16
    # ==========================================================
    # 1. enable_flash=True: 尝试使用 Flash Attention
    # 2. enable_math=False: 禁止使用导致 OOM 的传统数学计算
    # 3. enable_mem_efficient=True: 允许使用 xFormers 的省显存算法
    print("\n[Optimization] Enabling Flash Attention & FP16 to solve OOM...")
    
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            # Flash Attention 必须配合 FP16 或 BF16 使用
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                results = runner.test()
    except RuntimeError as e:
        print(f"\n[Warning] Flash Attention failed or not supported: {e}")
        print("[Retry] Falling back to standard FP16 (might OOM)...")
        # 如果 Flash Attention 失败（比如显卡不支持），退回到普通 FP16
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            results = runner.test()

    # 保存结果
    if runner.rank == 0:
        if results is None: results = {}
        print(f"\n[Evaluation Done] Results: {results}")
        results.update({'Model': cfg.model.type, 'Dataset': cfg.dataset_type})
        excel_name = f'results_{cfg.dataset_type}.xlsx'
        append_experiment_result(excel_name, [results])
        txt_path = os.path.join(cfg.work_dir, f'results_{cfg.dataset_type}.txt')
        with open(txt_path, 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"Result saved to {txt_path}")

if __name__ == '__main__':
    main()