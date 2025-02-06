import torch

def contrastive_loss(features, labels, temperature=0.07):
    """
    计算监督对比学习损失。
    
    Args:
        features: Tensor of shape (bs*num_tokens, embed_dim).
        labels: Tensor of shape (bs*num_tokens) with integer class labels.
        temperature: Temperature scaling factor for contrastive loss.
    
    Returns:
        loss: Supervised contrastive loss (scalar).
    """
    # Step 1: Compute pairwise similarity (dot product)
    similarity_matrix = torch.matmul(features, features.T)  # (bs * num_tokens, bs * num_tokens)
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    # Step 2: Create a mask to identify positive pairs
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (bs * num_tokens, bs * num_tokens)
    mask_self = torch.eye(label_mask.size(0), device=features.device)
    positive_mask = label_mask - mask_self  # Remove diagonal (self-pairs)

    # Step 3: Compute log probabilities for positive pairs
    log_prob = similarity_matrix / temperature - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Step 4: Sum log probabilities for positive samples
    positive_log_prob = positive_mask * log_prob
    positive_count = positive_mask.sum(dim=1)
    
    loss = -(positive_log_prob.sum(dim=1) / positive_count.clamp(min=1))  # Avoid division by zero
    return loss.mean()
