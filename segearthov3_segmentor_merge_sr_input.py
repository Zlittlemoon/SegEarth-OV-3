import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image
from torchvision import transforms

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class TTSTImageSR(torch.nn.Module):
    """4x TTST image super-resolution wrapper used before SAM3 image encoding."""

    def __init__(
        self,
        ckpt_path,
        input_size=256,
        sr_dim=180,
        device=torch.device("cuda"),
        allow_bicubic_fallback=False,
        use_official_ttst_init=True,
        use_official_ttst_eval=True,
        strict_sr_load=False,
        use_sr_amp=False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.scale = 4
        self.device = device
        self.allow_bicubic_fallback = bool(allow_bicubic_fallback)
        self.use_official_ttst_init = bool(use_official_ttst_init)
        self.use_official_ttst_eval = bool(use_official_ttst_eval)
        self.strict_sr_load = bool(strict_sr_load)
        self.use_sr_amp = bool(use_sr_amp)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.net = None

        ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        if ckpt_path is None or not ckpt_path.exists():
            if self.allow_bicubic_fallback:
                print(
                    "[SR INPUT] TTST checkpoint not found; using bicubic fallback: "
                    f"{ckpt_path}"
                )
                return
            raise FileNotFoundError(
                f"TTST checkpoint not found: {ckpt_path}. "
                "Set sr_ckpt_path to a valid 4x TTST checkpoint, or set "
                "allow_bicubic_fallback=True for pipeline debugging."
            )

        try:
            ttst_module = importlib.import_module("third_party.TTST.model_archs.TTST_arc")
            ttst_cls = getattr(ttst_module, "TTST")
        except (ImportError, AttributeError) as exc:
            if self.allow_bicubic_fallback:
                print(
                    "[SR INPUT] TTST implementation import failed; using bicubic fallback. "
                    f"Error: {exc}"
                )
                return
            raise ImportError(
                "Cannot import third_party.TTST.model_archs.TTST_arc.TTST. "
                "Put the TTST implementation under third_party/TTST/model_archs/TTST_arc.py, "
                "or set allow_bicubic_fallback=True for pipeline debugging."
            ) from exc

        if self.use_official_ttst_init:
            net = ttst_cls()
        else:
            net = ttst_cls(
                img_size=self.input_size,
                upscale=self.scale,
                embed_dim=int(sr_dim),
                upsampler="pixelshuffle",
            )
        if self.use_official_ttst_eval and self.device.type == "cuda":
            device_index = self.device.index if self.device.index is not None else 0
            self.net = torch.nn.DataParallel(net, device_ids=[device_index])
        else:
            self.net = net
        self._load_checkpoint(ckpt_path)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.to(self.device)

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if self.use_official_ttst_eval:
            try:
                self.net.load_state_dict(ckpt, strict=self.strict_sr_load)
                print(f"[SR INPUT] loaded TTST checkpoint with official eval flow: {ckpt_path}")
                print(
                    "[SR INPUT] "
                    f"official_init={self.use_official_ttst_init}, "
                    f"official_eval={self.use_official_ttst_eval}, "
                    f"strict_load={self.strict_sr_load}, "
                    f"use_amp={self.use_sr_amp}"
                )
                print("[SR INPUT] missing=0, unexpected=0")
                return
            except RuntimeError as exc:
                if self.strict_sr_load:
                    raise
                print(
                    "[SR INPUT] official eval checkpoint load failed; "
                    f"falling back to compatible load. Error: {exc}"
                )

        if isinstance(ckpt, dict):
            for key in ("params_ema", "params", "state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    print(f"[SR INPUT] using checkpoint key: {key}")
                    break
            else:
                state = ckpt
        else:
            state = ckpt

        for prefix in ("module.", "net.", "model.", "generator.", "G."):
            keys = list(state.keys()) if isinstance(state, dict) else []
            if keys and all(key.startswith(prefix) for key in keys):
                state = {key[len(prefix):]: value for key, value in state.items()}
                print(f"[SR INPUT] stripped checkpoint prefix: {prefix}")

        missing, unexpected = self.net.load_state_dict(state, strict=self.strict_sr_load)
        print(f"[SR INPUT] loaded TTST checkpoint: {ckpt_path}")
        print(
            "[SR INPUT] "
            f"official_init={self.use_official_ttst_init}, "
            f"official_eval={self.use_official_ttst_eval}, "
            f"strict_load={self.strict_sr_load}, "
            f"use_amp={self.use_sr_amp}"
        )
        print(f"[SR INPUT] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("[SR INPUT] first missing:", missing[:20])
        if unexpected:
            print("[SR INPUT] first unexpected:", unexpected[:20])

    @staticmethod
    def _pil_to_tensor(image):
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return tensor.float().div(255.0).unsqueeze(0)

    def _pil_to_tensor_official(self, image):
        return self.to_tensor(image).unsqueeze(0)

    @staticmethod
    def _tensor_to_pil_official(tensor):
        prediction = tensor.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)
        prediction = prediction.transpose(1, 2, 0)
        array = np.uint8(prediction)
        return Image.fromarray(array, mode="RGB")

    @torch.no_grad()
    def forward(self, image):
        if image.size != (self.input_size, self.input_size):
            image = image.resize((self.input_size, self.input_size), Image.BICUBIC)

        if self.net is None:
            sr_size = (self.input_size * self.scale, self.input_size * self.scale)
            return image.resize(sr_size, Image.BICUBIC)

        if self.use_official_ttst_eval:
            lr = self._pil_to_tensor_official(image)
        else:
            lr = self._pil_to_tensor(image)
        lr = lr.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and self.use_sr_amp,
        ):
            sr = self.net(lr)
        return self._tensor_to_pil_official(sr.float())


class RealESRGANImageSR:
    """Real-ESRGAN 4x image super-resolution wrapper before SAM3 image encoding."""

    @staticmethod
    def _patch_torchvision_functional_tensor_import():
        """Compat for BasicSR with newer torchvision versions.

        BasicSR imports rgb_to_grayscale from torchvision.transforms.functional_tensor,
        which was removed in recent torchvision releases. Provide that legacy module
        name before importing basicsr.
        """
        legacy_name = "torchvision.transforms.functional_tensor"
        if legacy_name in sys.modules:
            return

        from torchvision.transforms.functional import rgb_to_grayscale

        legacy_module = types.ModuleType(legacy_name)
        legacy_module.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[legacy_name] = legacy_module

    def __init__(
        self,
        model_name="RealESRGAN_x4plus",
        model_path="weights/Real-ESRGAN/RealESRGAN_x4plus.pth",
        scale=4,
        dni_weight=None,
        outscale=4,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None,
        allow_bicubic_fallback=False,
    ):
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path is not None else None
        self.scale = int(scale)
        self.outscale = float(outscale)
        self.allow_bicubic_fallback = bool(allow_bicubic_fallback)
        self.upsampler = None

        if self.model_path is None or not self.model_path.exists():
            if self.allow_bicubic_fallback:
                print(
                    "[SR INPUT] Real-ESRGAN checkpoint not found; using bicubic fallback: "
                    f"{self.model_path}"
                )
                return
            raise FileNotFoundError(
                f"Real-ESRGAN checkpoint not found: {self.model_path}. "
                "Download the matching Real-ESRGAN weight, or set "
                "allow_bicubic_fallback=True for pipeline debugging."
            )

        try:
            self._patch_torchvision_functional_tensor_import()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            if self.allow_bicubic_fallback:
                print(
                    "[SR INPUT] Real-ESRGAN import failed; using bicubic fallback. "
                    f"Error: {exc}"
                )
                return
            raise ImportError(
                "Cannot import Real-ESRGAN dependencies. Install the Real-ESRGAN "
                "package and basicsr, or set allow_bicubic_fallback=True."
            ) from exc

        if model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            netscale = 4
        else:
            raise ValueError(
                f"Unsupported Real-ESRGAN model_name={model_name}. "
                "Currently supported: RealESRGAN_x4plus."
            )

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(self.model_path),
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=gpu_id,
        )
        print(
            "[SR INPUT] loaded Real-ESRGAN checkpoint: "
            f"{self.model_path}, model_name={model_name}, outscale={self.outscale}, "
            f"tile={tile}, half={half}"
        )

    def __call__(self, image):
        if self.upsampler is None:
            sr_size = (image.size[0] * self.scale, image.size[1] * self.scale)
            return image.resize(sr_size, Image.BICUBIC)

        img_rgb = np.asarray(image.convert("RGB"))
        # Official Real-ESRGAN inference reads images with cv2, so the enhancer
        # expects BGR channel order and returns BGR.
        img_bgr = img_rgb[:, :, ::-1]
        output_bgr, _ = self.upsampler.enhance(img_bgr, outscale=self.outscale)
        output_rgb = output_bgr[:, :, ::-1]
        return Image.fromarray(output_rgb, mode="RGB")


@MODELS.register_module()
class SegEarthOV3SegmentationSRInput(BaseSegmentor):
    def __init__(
        self,
        classname_path,
        device=torch.device("cuda"),
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        sam3_checkpoint_path="weights/sam3/sam3.pt",
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        use_sr_input=True,
        sr_backend="ttst",
        sr_ckpt_path="weights/TTST/ttst_4x.pth",
        sr_input_size=256,
        sr_dim=180,
        allow_bicubic_fallback=False,
        use_official_ttst_init=True,
        use_official_ttst_eval=True,
        strict_sr_load=False,
        use_sr_amp=False,
        realesrgan_model_name="RealESRGAN_x4plus",
        realesrgan_model_path="weights/Real-ESRGAN/RealESRGAN_x4plus.pth",
        realesrgan_outscale=4,
        realesrgan_tile=0,
        realesrgan_tile_pad=10,
        realesrgan_pre_pad=0,
        realesrgan_half=True,
        realesrgan_gpu_id=None,
        keep_sr_output=False,
        use_original_logit_merge=False,
        debug_sr=False,
        debug_sr_dir="resources/debug_sr_input_isaid_split",
        debug_sr_log_path=None,
        debug_sr_max_save=5,
        **kwargs,
    ):
        super().__init__()

        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=sam3_checkpoint_path,
            device=str(device),
        )
        self.processor = Sam3Processor(
            model,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        self.use_sr_input = bool(use_sr_input)
        self.sr_backend = sr_backend.lower().strip()
        self.keep_sr_output = bool(keep_sr_output)
        self.use_original_logit_merge = bool(use_original_logit_merge)
        self.debug_sr = bool(debug_sr)
        self.debug_sr_dir = Path(debug_sr_dir)
        self.debug_sr_log_path = (
            Path(debug_sr_log_path)
            if debug_sr_log_path is not None
            else self.debug_sr_dir / "sr_debug.log"
        )
        self.debug_sr_max_save = int(debug_sr_max_save)
        self._debug_sr_saved = 0
        self.sr_input_size = int(sr_input_size)
        self.sr_scale = 4
        self.sr_model = None
        if self.use_sr_input:
            if self.sr_backend == "ttst":
                self.sr_model = TTSTImageSR(
                    ckpt_path=sr_ckpt_path,
                    input_size=self.sr_input_size,
                    sr_dim=sr_dim,
                    device=device,
                    allow_bicubic_fallback=allow_bicubic_fallback,
                    use_official_ttst_init=use_official_ttst_init,
                    use_official_ttst_eval=use_official_ttst_eval,
                    strict_sr_load=strict_sr_load,
                    use_sr_amp=use_sr_amp,
                )
            elif self.sr_backend == "realesrgan":
                self.sr_model = RealESRGANImageSR(
                    model_name=realesrgan_model_name,
                    model_path=realesrgan_model_path,
                    scale=4,
                    outscale=realesrgan_outscale,
                    tile=realesrgan_tile,
                    tile_pad=realesrgan_tile_pad,
                    pre_pad=realesrgan_pre_pad,
                    half=realesrgan_half,
                    gpu_id=realesrgan_gpu_id,
                    allow_bicubic_fallback=allow_bicubic_fallback,
                )
                self.sr_input_size = None
            elif self.sr_backend == "bicubic":
                self.sr_model = None
            else:
                raise ValueError(
                    f"Unsupported sr_backend={sr_backend}. "
                    "Expected one of: 'ttst', 'realesrgan', 'bicubic'."
                )

        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

    def _super_resolve_image(self, image, image_path=None):
        if not self.use_sr_input:
            return image
        original_image = image
        if self.sr_backend == "bicubic":
            expected_size = (image.size[0] * self.sr_scale, image.size[1] * self.sr_scale)
            sr_image = image.resize(expected_size, Image.BICUBIC)
        else:
            sr_image = self.sr_model(image)
            if self.sr_backend == "ttst":
                expected_size = (
                    self.sr_input_size * self.sr_scale,
                    self.sr_input_size * self.sr_scale,
                )
            else:
                expected_size = (
                    image.size[0] * self.sr_scale,
                    image.size[1] * self.sr_scale,
                )
        if sr_image.size != expected_size:
            sr_image = sr_image.resize(expected_size, Image.BICUBIC)

        if self.debug_sr and self._debug_sr_saved < self.debug_sr_max_save:
            self.debug_sr_dir.mkdir(parents=True, exist_ok=True)
            self.debug_sr_log_path.parent.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem if image_path is not None else f"sample_{self._debug_sr_saved}"
            bicubic_image = original_image.resize(expected_size, Image.BICUBIC)
            sr_arr = np.asarray(sr_image).astype(np.float32)
            bicubic_arr = np.asarray(bicubic_image).astype(np.float32)
            bicubic_mae = float(np.mean(np.abs(sr_arr - bicubic_arr)))
            bicubic_max_diff = float(np.max(np.abs(sr_arr - bicubic_arr)))

            original_image.save(
                self.debug_sr_dir
                / f"{stem}_original_{original_image.size[0]}x{original_image.size[1]}.png"
            )
            bicubic_image.save(
                self.debug_sr_dir
                / f"{stem}_bicubic_{bicubic_image.size[0]}x{bicubic_image.size[1]}.png"
            )
            sr_image.save(
                self.debug_sr_dir / f"{stem}_sr_{sr_image.size[0]}x{sr_image.size[1]}.png"
            )
            debug_message = (
                "[SR INPUT DEBUG] "
                f"image={image_path}, original_size={original_image.size}, "
                f"sam3_input_size={sr_image.size}, "
                f"sr_backend={self.sr_backend}, "
                f"mae_vs_bicubic={bicubic_mae:.4f}, "
                f"max_diff_vs_bicubic={bicubic_max_diff:.1f}"
            )
            with open(self.debug_sr_log_path, "a", encoding="utf-8") as f:
                f.write(debug_message + "\n")
            self._debug_sr_saved += 1
        return sr_image

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        if self.use_original_logit_merge:
            seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        else:
            # SAM3 semantic logits are raw signed logits. Initializing with 0 would
            # clamp all negative class responses to 0, and argmax ties would collapse
            # large regions to class 0. Use -inf so every query keeps its real score.
            seg_logits = torch.full(
                (self.num_queries, h, w),
                float("-inf"),
                device=self.device,
            )

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=query_word,
                )

                if self.use_transformer_decoder:
                    if inference_state["masks_logits"].shape[0] > 0:
                        inst_len = inference_state["masks_logits"].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state["masks_logits"][inst_id].squeeze()
                            instance_score = inference_state["object_score"][inst_id]

                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape),
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(
                                seg_logits[query_idx],
                                instance_logits * instance_score,
                            )

                if self.use_sem_seg:
                    semantic_logits = inference_state["semantic_mask_logits"]
                    if semantic_logits.dim() == 2:
                        semantic_logits_4d = semantic_logits.unsqueeze(0).unsqueeze(0)
                    elif semantic_logits.dim() == 3:
                        semantic_logits_4d = semantic_logits.unsqueeze(1)
                    else:
                        semantic_logits_4d = semantic_logits

                    if semantic_logits_4d.shape[-2:] != (h, w):
                        semantic_logits_4d = F.interpolate(
                            semantic_logits_4d,
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )
                    semantic_logits = semantic_logits_4d.squeeze()

                    if self.use_original_logit_merge:
                        seg_logits[query_idx] = torch.max(
                            seg_logits[query_idx],
                            semantic_logits,
                        )
                    elif not self.use_transformer_decoder:
                        seg_logits[query_idx] = semantic_logits
                    else:
                        seg_logits[query_idx] = torch.max(
                            seg_logits[query_idx],
                            semantic_logits,
                        )

                if (not self.use_original_logit_merge) and torch.isinf(seg_logits[query_idx]).any():
                    seg_logits[query_idx] = torch.nan_to_num(
                        seg_logits[query_idx],
                        neginf=-1.0e4,
                    )

                if self.use_presence_score:
                    seg_logits[query_idx] = (
                        seg_logits[query_idx] * inference_state["presence_score"]
                    )

        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)

                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."

        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get("img_path")
            image = Image.open(image_path).convert("RGB")
            ori_shape = meta["ori_shape"]

            sam3_image = self._super_resolve_image(image, image_path=image_path)

            if self.slide_crop > 0 and (
                self.slide_crop < sam3_image.size[0] or self.slide_crop < sam3_image.size[1]
            ):
                seg_logits = self.slide_inference(sam3_image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(sam3_image)

            target_shape = seg_logits.shape[-2:] if self.keep_sr_output else ori_shape
            if seg_logits.shape[-2:] != target_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)

            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
                }
            )

        return data_samples

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(",")
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "") for item in class_names]
    return class_names, class_indices
