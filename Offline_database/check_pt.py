import torch

# 替换成你实际的 pt 文件路径
pt_path = "Offline_database/Million-AID_train_image_features.pt" 

try:
    print(f"正在加载 {pt_path} ...")
    data = torch.load(pt_path, map_location='cpu')
    
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print("好消息！这是一个字典。包含的键有：", data.keys())
        
        # 检查是否有路径列表
        if 'image_paths' in data:
            print(f"找到路径列表！长度: {len(data['image_paths'])}")
            print("前3个路径示例:", data['image_paths'][:3])
        elif 'names' in data:
            print(f"找到名称列表！长度: {len(data['names'])}")
        else:
            print("坏消息：字典里好像没有路径键。")
            
        # 检查特征
        if 'features' in data:
            print(f"特征形状: {data['features'].shape}")
        elif 'embeddings' in data:
             print(f"特征形状: {data['embeddings'].shape}")
             
    elif isinstance(data, torch.Tensor):
        print("坏消息：这只是一个纯 Tensor，不包含路径信息。")
        print(f"形状: {data.shape}")
        print("这意味着路径列表还在原来的脚本代码里，或者在另一个文件里。")
        
except Exception as e:
    print(f"读取出错: {e}")