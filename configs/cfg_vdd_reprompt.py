_base_ = './base_config.py'

# model settings
model = dict(
    type='RSGPNetSegmentation',
    classname_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
    confidence_threshold=0.5,
    # slide_stride=512,
    # slide_crop=512,
)

# dataset settings
dataset_type = 'VDDDataset'
data_root = 'data/VDD'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/src',
            seg_map_path='test/gt'),
        pipeline=test_pipeline))

test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(
        type='BoundaryF1Metric',
        num_classes=7,
        ignore_index=255,
        epsilon=0.02
    )
]
custom_imports = dict(imports=['boundary_f1_metric'], allow_failed_imports=False)