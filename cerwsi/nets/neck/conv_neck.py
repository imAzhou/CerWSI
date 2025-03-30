import torch
from torch import nn
import math

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeck(nn.Module):
    def __init__(self, args):
        
        super(ConvNeck, self).__init__()

        input_dim = args.backbone_cfg['backbone_output_dim'][0]
        out_chans = args.neck_output_dim[0]
        self.conv_module = nn.Sequential(
            nn.Conv2d(
                input_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        
    def forward(self, feature_emb: torch.Tensor):
        '''
        Args:
            feature_emb: (bs, L, input_dim)
        Return:
            feature_emb: (bs, L, out_chans)
        '''
        # feature_emb: (bs, L, input_dim)
        bs, L, input_dim = feature_emb.shape
        H = W = int(math.sqrt(L))
        feature_emb = feature_emb.reshape(bs, input_dim, H, W)  # (bs, input_dim, H, W)
        feature_emb = self.conv_module(feature_emb)  # (bs, out_chans, H, W)
        feature_emb = feature_emb.reshape(bs, L, -1)  # 变回 (bs, L, out_chans)
        return feature_emb
    
