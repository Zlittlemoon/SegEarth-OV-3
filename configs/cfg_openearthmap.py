_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    use_instance_score=True,
    use_scale_norm=True,
    inst_temp=10.0,
    sem_temp=1.0, 
    slide_stride=512,
    slide_crop=512,
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = 'data/OpenEarthMap'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))