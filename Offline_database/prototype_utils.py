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
        
        ckpt = torch.load(clip_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
        self.clip_model = model.to(device).eval()
        
        # 2. 加载离线特征库
        print(f"[PrototypeManager] Loading Database: {database_path}...")
        data = torch.load(database_path, map_location="cpu") 
        
        # [修改点] 确保数据库特征是 float32 (为了检索精度) 或者转为和 device 一致
        self.db_features = data["image_features"].to(device).float() 
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
            similarity = text_feat @ self.db_features.T
            
            # 3. 取 Top-K
            topk_scores, topk_indices = similarity.topk(topk, dim=1)
            
            # 4. 获取特征 (Batch化处理)
            # 使用高级索引直接提取，比 for 循环更快
            # topk_indices: [B, K] -> flatten -> [B*K]
            indices_flat = topk_indices.view(-1)
            v_ref_flat = self.db_features[indices_flat] # [B*K, 512]
            v_ref = v_ref_flat.view(len(text_prompts), topk, 512) # [B, K, 512]
            
            return v_ref, topk_scores