# base config (soft-prompt)

model = dict(
    type='SegEarthOV3SegmentationSoftPrompt',
    model_type='SAM3',
    checkpoint_path='weights/sam3/sam3.pt',
    finetuned_checkpoint_path='outputs/dlrsd_v0a_seghead_only_4x4090/checkpoints/checkpoint.pt',
    use_soft_prompt=True,
    bpe_path='./sam3/assets/bpe_simple_vocab_16e6.txt.gz',
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='val')

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
