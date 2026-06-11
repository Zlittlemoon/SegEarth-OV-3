_base_ = "./base_config_sr_input.py"

# model settings
model = dict(
    classname_path="./configs/cls_iSAID_split.txt",
    prob_thd=0.0,
    confidence_threshold=0.4,
    bg_idx=255,
    use_sr_input=True,
    sr_backend="realesrgan",
    realesrgan_model_name="RealESRGAN_x4plus",
    realesrgan_model_path="weights/Real-ESRGAN/RealESRGAN_x4plus.pth",
    realesrgan_outscale=4,
    realesrgan_tile=0,
    realesrgan_tile_pad=10,
    realesrgan_pre_pad=0,
    realesrgan_half=True,
    realesrgan_gpu_id=None,
    allow_bicubic_fallback=False,
    keep_sr_output=False,
    use_original_logit_merge=True,
)

# dataset settings
dataset_type = "iSAIDDataset"
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
            img_path="val/imgs",
            seg_map_path="val/D2masks",
        ),
        pipeline=test_pipeline,
    ),
)
