import os
import os.path as osp
import argparse
import openpyxl
import torch
from mmengine.runner import Runner
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info
from datetime import timedelta # 引入 timedelta

from model.gcsam3_segmentor import SegEarthOV3Segmentation
import custom_datasets  # 确保自定义数据集被注册


def parse_args():
    parser = argparse.ArgumentParser(
        description='CorrCLIP evaluation with MMSeg')
    parser.add_argument('config', default='./configs/cfg_loveda.py')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='./show_dir/',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    
    # [关键参数] 启动器选择
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    # torchrun 会自动传递 local_rank，这是为了兼容性
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
        sheet['A1'] = 'Model'
        sheet['B1'] = 'Dataset'
        sheet['C1'] = 'aAcc'
        sheet['D1'] = 'mIoU'
        sheet['E1'] = 'mAcc'

    last_row = sheet.max_row

    for index, result in enumerate(experiment_data, start=1):
        sheet.cell(row=last_row + index, column=1, value=result.get('Model', 'N/A'))
        sheet.cell(row=last_row + index, column=2, value=result.get('Dataset', 'N/A'))
        sheet.cell(row=last_row + index, column=3, value=result.get('aAcc', 0))
        sheet.cell(row=last_row + index, column=4, value=result.get('mIoU', 0))
        sheet.cell(row=last_row + index, column=5, value=result.get('mAcc', 0))

    workbook.save(file_path)


def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    
    # 配置输出目录
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    # [新增] 核心修复：强制修改分布式环境配置
    # 增加 timeout 到 7200秒 (2小时)，防止大图推理时 NCCL 超时
    cfg.env_cfg = dict(
        cudnn_benchmark=True,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl', timeout=timedelta(seconds=7200)) 
    )
    # ----------------------------------------------------------
    # [配置模型权重]
    # ----------------------------------------------------------
    # 请根据实际情况修改这个路径
    trained_adapter_path = "weights/Potsdam_ddp_adapter_epoch_20.pth"
    
    if os.path.exists(trained_adapter_path):
        cfg.model.adapter_ckpt_path = trained_adapter_path
    else:
        # 仅在 Rank 0 打印警告，避免多卡刷屏
        if args.local_rank == 0:
            print(f"Warning: Adapter checkpoint not found at {trained_adapter_path}!")

    # ----------------------------------------------------------
    # [初始化 Runner]
    # ----------------------------------------------------------
    # Runner 会自动处理分布式环境初始化 (init_dist)
    runner = Runner.from_cfg(cfg)

    # ----------------------------------------------------------
    # [开始测试]
    # ----------------------------------------------------------
    # Runner 会自动处理多卡数据切分和结果收集
    results = runner.test()

    # ----------------------------------------------------------
    # [结果保存 - 仅限主进程]
    # ----------------------------------------------------------
    # 在多卡模式下，非主进程的 results 可能是 None 或空，必须检查 rank
    if runner.rank == 0:
        if results is None:
            results = {}
            
        print(f"\n[Evaluation Done] Results: {results}")

        results.update({
            'Model': cfg.model.type,
            'Dataset': cfg.dataset_type
        })

        # 1. 保存到 Excel
        excel_name = f'results_{cfg.dataset_type}.xlsx'
        append_experiment_result(excel_name, [results])
        print(f"Result saved to {excel_name}")

        # 2. 保存到 txt
        txt_path = os.path.join(cfg.work_dir, f'results_{cfg.dataset_type}.txt')
        with open(txt_path, 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"Result saved to {txt_path}")


if __name__ == '__main__':
    main()