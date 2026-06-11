_base_ = "./base_config_sr_input.py"

model = dict(
    classname_path="./configs/cls_iSAID.txt",
    prob_thd=0.5,
    confidence_threshold=0.4,
    use_sr_input=True,
    sr_ckpt_path="weights/TTST/ttst_4x.pth",
    sr_input_size=256,
    sr_dim=180,
    allow_bicubic_fallback=False,
    keep_sr_output=False,
    slide_stride=0,
    slide_crop=0,
)

dataset_type = "iSAIDDataset"
data_root = "data/isaid_patch"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=test_pipeline,
    ),
)
