import torch
import math

# 假设 feature_emb 形状为 (bs, L, input_dim)
bs, L, input_dim = 2, 16, 8  # 例如 L=16, 这里 L 需要是 H × W 的完全平方数
H = W = int(math.sqrt(L))  # 计算 H 和 W

# 随机生成 feature_emb
feature_emb = torch.randn(bs, L, input_dim)  # (bs, L, input_dim)

# 直接 reshape，不使用 permute
feature_emb_reshaped = feature_emb.permute(0, 2, 1).reshape(bs, input_dim, H, W)  # 变成 (bs, input_dim, H, W)

print("Original shape:", feature_emb.shape)  # (bs, L, input_dim)
print("Reshaped shape:", feature_emb_reshaped.shape)  # (bs, input_dim, H, W)