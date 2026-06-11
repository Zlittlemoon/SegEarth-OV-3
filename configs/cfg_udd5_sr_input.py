_base_ = "./base_config_sr_input.py"

# model settings
model = dict(
    classname_path="./configs/cls_udd5.txt",
    confidence_threshold=0.5,
    prob_thd=0.1,
    bg_idx=4,
    use_sr_input=True,
    sr_ckpt_path="weights/TTST/ttst_4x.pth",
    sr_input_size=256,
    sr_dim=180,
    allow_bicubic_fallback=False,
    keep_sr_output=False,
    use_original_logit_merge=True,
    # slide_stride=512,
    # slide_crop=512,
)

# dataset settings
dataset_type = "UDD5Dataset"
data_root = "data/UDD5"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Resize", scale=(448, 448), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="val/src",
            seg_map_path="val/gt",
        ),
        pipeline=test_pipeline,
    ),
)
