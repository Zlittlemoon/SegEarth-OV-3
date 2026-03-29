_base_ = './base_config.py'

model = dict(
    type='SegEarthOV3SoftPromptSegmentation',
    classname_path='./configs/cls_loveda.txt',
    sam3_base_ckpt='weights/sam3/sam3.pt',
    soft_prompt_ckpt='outputs/dlrsd_prompt_soft_only_4x3090/checkpoints/checkpoint_10.pt',
    confidence_threshold=0.5,
    prob_thd=0.0,
    slide_stride=0,
    slide_crop=0,
    device='cuda',
)

dataset_type = 'LoveDADataset'
data_root = 'data/LoveDA'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'
        ),
        pipeline=test_pipeline
    )
)