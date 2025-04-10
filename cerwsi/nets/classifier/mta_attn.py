import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTokenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, c_q=3, c_k=3, c_h=2):
        """
        图像任务专用MTA (无因果掩码)
        Args:
            embed_dim: 输入特征维度
            num_heads: 注意力头数
            c_q: 查询方向的卷积核大小
            c_k: 键方向的卷积核大小
            c_h: 头混合分组大小
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_h = c_h
        
        # Key-Query投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key-Query卷积 (分组卷积实现)
        self.conv2d = nn.Conv2d(
            in_channels=num_heads,
            out_channels=num_heads,
            kernel_size=(c_q, c_k),
            padding=(c_q // 2, c_k // 2),  # 保持空间尺寸不变
            groups=num_heads,  # 每个头独立处理
            bias=False
        )
        
        # 头混合参数 (每组c_h个头)
        assert num_heads % c_h == 0, "num_heads必须能被c_h整除"
        self.head_mix = nn.Parameter(torch.eye(c_h).repeat(num_heads // c_h, 1, 1))
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        """
        输入:
            query: (B, H_q, W_q, C) 或 (B, N_q, C)  
            key: (B, H_k, W_k, C) 或 (B, N_k, C)
            value: 同key形状
        输出:
            (B, H_q, W_q, C) 或 (B, N_q, C)
        """
        # 处理2D/1D输入统一化
        if query.dim() == 4:
            B, H_q, W_q, C = query.shape
            query = query.view(B, H_q * W_q, C)
            key = key.view(B, -1, C)
            value = value.view(B, -1, C)
            is_2d = True
        else:
            B, N_q, C = query.shape
            is_2d = False
        
        # 线性投影
        Q = self.q_proj(query)  # (B, N_q, C)
        K = self.k_proj(key)    # (B, N_k, C)
        V = self.v_proj(value)  # (B, N_k, C)
        
        # 分割多头
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, N_q, hd)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, N_k, hd)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, N_k, hd)
        
        # 计算标准注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, N_q, N_k)
        
        # Key-Query卷积 (无因果掩码)
        conv_attn = self.conv2d(attn_scores)  # (B, nh, N_q, N_k)
        
        # Softmax归一化
        attn_weights = F.softmax(conv_attn, dim=-1)
        
        # Head Mixing (分组头混合)
        attn_weights = attn_weights.view(B, -1, self.c_h, attn_weights.size(-2), attn_weights.size(-1))  # (B, ng, c_h, N_q, N_k)
        mixed_weights = torch.einsum('bghij,ghk->bgkij', attn_weights, self.head_mix)  # (B, ng, c_h, N_q, N_k)
        mixed_weights = mixed_weights.reshape(B, self.num_heads, attn_weights.size(-2), attn_weights.size(-1))
        
        # 加权求和
        output = torch.matmul(mixed_weights, V)  # (B, nh, N_q, hd)
        output = output.transpose(1, 2).reshape(B, -1, self.embed_dim)  # (B, N_q, C)
        
        # 输出投影
        output = self.out_proj(output)
        
        # 恢复2D形状 (如果是图像输入)
        if is_2d:
            output = output.view(B, H_q, W_q, -1)
        
        return output
    