import torch
import open_clip

class PrototypeManager:
    def __init__(self, 
                 clip_model_name="ViT-B-32", 
                 clip_ckpt_path="weights/RemoteCLIP/RemoteCLIP-ViT-B-32.pt", 
                 database_path="Offline_database/Million-AID_train_image_features.pt",
                 device="cuda"):
        
        self.device = device
        print(f"[PrototypeManager] Loading RemoteCLIP: {clip_model_name}...")
        
        # 1. 加载 RemoteCLIP
        model, _, _ = open_clip.create_model_and_transforms(clip_model_name)
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # safer load (avoid FutureWarning): if ckpt only contains tensors/weights
        ckpt = torch.load(clip_ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        self.clip_model = model.to(device).eval()
        
        # 2. 加载离线特征库
        print(f"[PrototypeManager] Loading Database: {database_path}...")
        data = torch.load(database_path, map_location="cpu")
        # ensure float32 and L2-normalize each vector for stable similarity scale
        self.db_features = data["image_features"].to(torch.float32)
        self.db_features = self.db_features / (self.db_features.norm(dim=1, keepdim=True) + 1e-12)
        self.db_features = self.db_features.to(device)
        self.db_paths = data["image_paths"]
        print(f"[PrototypeManager] Loaded {len(self.db_paths)} images.")

    def get_prototypes(self, text_prompts, topk=5):
        """
        输入: text_prompts (list of str)
        输出: v_ref [B, K, 512], scores [B, K]
        """
        with torch.no_grad():
            # 1. 文本编码
            text_tokens = self.tokenizer(text_prompts).to(self.device)
            text_feat = self.clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
            # [修改点] 确保类型匹配
            text_feat = text_feat.to(self.db_features.dtype)

            # 2. 相似度计算
            # compute cosine-like similarity (both sides normalized)
            text_feat = text_feat / (text_feat.norm(dim=1, keepdim=True) + 1e-12)
            sims = (text_feat @ self.db_features.T).cpu()  # in [-1,1]
            topk_vals, topk_idx = sims.topk(topk, dim=-1)
            # normalize topk scores to [0,1] via softmax for stable weighting
            proto_scores = torch.softmax(topk_vals, dim=-1)
            # gather prototypes and return (ensure device/dtype)
            prototypes = self.db_features[topk_idx].to(text_feat.device)  # [B, topk, dim]
            return prototypes, proto_scores