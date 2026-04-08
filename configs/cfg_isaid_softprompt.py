_base_ = './base_config_softprompt.py'

# model settings
model = dict(
    classname_path='./configs/cls_iSAID.txt',
    prob_thd=0.5,
    confidence_threshold=0.4,
    slide_stride=512,
    slide_crop=512,
    # Optional override (already set in base_config_softprompt.py)
    finetuned_checkpoint_path='outputs/dlrsd_prompt_soft_only_1xA6000_test_epoch20/checkpoints/checkpoint_20.pt',
    use_soft_prompt=True,
)

# dataset settings
dataset_type = 'iSAIDDataset'
data_root = 'data/isaid_patch'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
