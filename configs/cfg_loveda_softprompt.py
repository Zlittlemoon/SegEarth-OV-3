_base_ = './base_config_softprompt.py'

model = dict(
    classname_path='./configs/cls_loveda.txt',
    confidence_threshold=0.5,
    prob_thd=0.5,
    # finetuned_checkpoint_path='outputs/dlrsd_detector_presence_only_instance_maxlogit_nosoft_4x3090_lr_1e-5_batch2/checkpoints/checkpoint_20.pt',
    use_soft_prompt=True,
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