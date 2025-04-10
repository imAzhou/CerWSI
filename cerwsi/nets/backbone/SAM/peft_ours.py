import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from .attn_module import add_decomposed_rel_pos

class LoRALinear(nn.Module):
    """LoRA applied to a linear layer with learnable rank decomposition."""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank  # Rescale factor to balance the adaptation

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class DTCWTModule(nn.Module):
    """Multi-head Attention block with relative position embeddings and LoRA."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        feat_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.complex_weight_ll = nn.Parameter(torch.randn(dim, feat_size, feat_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_heads, head_dim, head_dim, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_heads, head_dim, head_dim, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_heads, head_dim,  dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_heads, head_dim,  dtype=torch.float32) * 0.02)
        
        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0 

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        origin_x = input_x

        B, H, W, C = input_x.shape
        input_x = input_x.permute(0,3,1,2)
        xl,xh = self.xfm(input_x)
        xl = xl * self.complex_weight_ll
        xh[0]=torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_heads, -1)

        x_real=xh[0][0]
        x_imag=xh[0][1]

        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], C, xh[0].shape[6])
        xh[0]=torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

        dtcwt_x = self.ifm((xl,xh)) # (bs, C, h, w)
        dtcwt_x = dtcwt_x.permute(0,2,3,1)

        output_x = origin_x + dtcwt_x
        return output_x
   

class AttentionDTCWT(nn.Module):
    """Multi-head Attention block with relative position embeddings and LoRA."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        feat_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

        self.dtcwt_module = DTCWTModule(dim, num_heads, feat_size)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        origin_x = self.proj(x)

        dtcwt_x = self.dtcwt_module(input_x)
        output_x = origin_x + dtcwt_x
        return output_x
    