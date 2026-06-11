_base_ = "./base_config_sr_input.py"

model = dict(
    classname_path="./configs/cls_iSAID_split.txt",
    prob_thd=0.0,
    confidence_threshold=0.4,
    bg_idx=255,
    use_sr_input=True,
    sr_backend="ttst",
    sr_ckpt_path="weights/TTST/ttst_4x.pth",
    sr_input_size=256,
    sr_dim=180,
    allow_bicubic_fallback=False,
    keep_sr_output=False,
    use_original_logit_merge=True,
)

dataset_type = "iSAIDSplitDataset"
data_root = "data/iSAID_split"

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
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        data_prefix=dict(
            img_path="val/images",
            seg_map_path="val/D2masks",
        ),
        pipeline=test_pipeline,
    ),
)