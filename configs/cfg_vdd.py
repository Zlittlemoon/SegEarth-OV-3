_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
    confidence_threshold=0.5,
    slide_stride=512,
    slide_crop=512,
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