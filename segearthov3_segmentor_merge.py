import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 sam3_infer_mode='serial',  # 'serial' (default, unchanged) or 'batch'
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
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
        assert sam3_infer_mode in ('serial', 'batch')
        self.sam3_infer_mode = sam3_infer_mode

    def _inference_single_view_batch(self, image):
        """Batched (one forward_grounding for all prompts) three-head fusion.

        Mirrors the serial path's fusion exactly, but runs the grounding once
        with img_ids=[0]*N, text_ids=[0..N-1].  The instance head needs the
        per-prompt confidence (keep) filter applied row by row, because each
        prompt keeps a different number of instances — this is the part that
        the semantic-only check did NOT exercise.
        """
        w, h = image.size
        n = self.num_queries
        seg_logits = torch.zeros((n, h, w), device=self.device)
        thr = self.processor.confidence_threshold

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            text_outputs = self.processor.model.backbone.forward_text(
                self.query_words, device=self.device)
            state["backbone_out"].update(text_outputs)
            geometric_prompt = self.processor.model._get_dummy_prompt(num_prompts=n)
            find_stage = FindStage(
                img_ids=torch.zeros(n, device=self.device, dtype=torch.long),
                text_ids=torch.arange(n, device=self.device, dtype=torch.long),
                input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                input_points=None, input_points_mask=None,
            )
            outputs = self.processor.model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )

            pred_logits = outputs["pred_logits"]            # (N, Q, 1)
            pred_masks = outputs["pred_masks"]              # (N, Q, h', w')
            presence = outputs["presence_logit_dec"].sigmoid()  # (N, 1)
            # (N, Q): object prob = sigmoid(logit) * presence, matches serial
            obj_probs = (pred_logits.sigmoid() * presence.unsqueeze(1)).squeeze(-1)

            sem_all = None
            if self.use_sem_seg:
                sem_all = interpolate(
                    outputs["semantic_seg"], (h, w),
                    mode="bilinear", align_corners=False).sigmoid().squeeze(1)  # (N,h,w)

            for qi in range(n):
                if self.use_transformer_decoder:
                    keep = obj_probs[qi] > thr                 # (Q,)
                    if keep.any():
                        masks = pred_masks[qi][keep]           # (K, h', w')
                        masks = interpolate(
                            masks.unsqueeze(1), (h, w),
                            mode="bilinear", align_corners=False).sigmoid().squeeze(1)  # (K,h,w)
                        scores = obj_probs[qi][keep]           # (K,)
                        # max over instances of (mask * score); zeros base is a no-op
                        folded = (masks * scores[:, None, None]).max(0)[0]
                        seg_logits[qi] = torch.max(seg_logits[qi], folded)

                if self.use_sem_seg:
                    seg_logits[qi] = torch.max(seg_logits[qi], sem_all[qi])

                if self.use_presence_score:
                    seg_logits[qi] = seg_logits[qi] * presence[qi].squeeze()

        return seg_logits

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        if self.sam3_infer_mode == 'batch':
            return self._inference_single_view_batch(image)
            
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                
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
        
        # Initialize accumulators
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
                
                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))
                
                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                
                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
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
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices