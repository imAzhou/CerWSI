import torch
from cerwsi.nets.classifier.mta_attn import MultiTokenAttention

attn = MultiTokenAttention(embed_dim=512, num_heads=8, c_q=5, c_k=5)
query = torch.randn(2, 100, 512)
key = value = torch.randn(2, 200, 512)
output = attn(query, key, value)
