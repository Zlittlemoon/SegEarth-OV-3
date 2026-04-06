_base_ = './base_config_softprompt.py'

# model settings
model = dict(
    classname_path='./configs/cls_vaihingen.txt',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
    finetuned_checkpoint_path='outputs/dlrsd_prompt_head_prompt_stage1_t05_1xA6000/checkpoints/checkpoint_20.pt',
    use_soft_prompt=True,
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = 'data/Vaihingen'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(448, 448), keep_ratio=True),
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
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
