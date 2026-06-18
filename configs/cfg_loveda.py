_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_loveda.txt',
    confidence_threshold=0.5,
    prob_thd=0.5,
    use_instance_score=True,
    use_scale_norm=True,
    inst_temp=10.0,
    sem_temp=1.0,   
    # slide_stride=512,
    # slide_crop=512,
)

# dataset settings
# Category labels: background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. 
# And the no-data regions were assigned 0 which should be ignored. 
dataset_type = 'LoveDADataset'
data_root = 'data/LoveDA'

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
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))