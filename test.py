import torch

# 假设 balanced_mask 是一个二维张量，形状为 (bs, num_tokens)
balanced_mask = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])  # 示例数据

# 假设 img_tokens 是一个三维张量，形状为 (bs, num_tokens, C)
img_tokens = torch.randn(2, 4, 3)  # 示例数据

# 假设 feat_gt 是一个二维张量，形状为 (bs, num_tokens)
feat_gt = torch.randn(2, 4)  # 示例数据

# 找到 balanced_mask 中值为 1 的索引
indices = balanced_mask.nonzero(as_tuple=True)

# 使用索引提取 img_tokens 和 feat_gt 中对应的值
selected_img_tokens = img_tokens[indices]
selected_feat_gt = feat_gt[indices]

print("Selected img_tokens:", selected_img_tokens)
print("Selected feat_gt:", selected_feat_gt)