import torch

# 假设 logits_tensor 是 (num_classes, num_tokens) 的概率值 tensor
logits_tensor = torch.tensor([
    [0.1, 0.7, 0.2],  # 类别 0
    [0.6, 0.8, 0.4],  # 类别 1
    [0.3, 0.4, 0.1],  # 类别 2
])

gt_tensor = torch.tensor([
    [0, 0, 0],  # 类别 0
    [0, 1, 1],  # 类别 1
    [1, 1, 1],  # 类别 2
])

# 计算每个类别中 token > 0.5 的数量
class_has_high_prob = (gt_tensor > 0.5).any(dim=1)

# 获取类别索引
class_indices = torch.nonzero(class_has_high_prob, as_tuple=False).squeeze(1)

# 输出类别索引
print(class_indices)