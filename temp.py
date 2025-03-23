import torch
import torch.nn as nn

class FourierFT(nn.Module):
    def __init__(
        self,
        n: int = 100, # number of trainable parameters
        alpha: float = 300.0, # scaling
        d1: int = 4096, # input dimension
        d2: int = 4096, # output dimension
        base_layer: nn.Module = nn.Linear # pre-trained layer
    ): # definitions
        super(FourierFT, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.n = n
        self.alpha = alpha
        self.base_layer = base_layer
        # entry initialization (no frequency bias)
        self.E = torch.randperm(d1 * d2)[:n]    # shape: (n, )
        self.E = torch.stack([self.E // self.d1, self.E % self.d2], dim=0)    # shape: (2, n)
        # spectral coefficient initialization
        self.c = nn.Parameter(torch.randn(n), requires_grad=True)    # shape: (n, )
    
    def forward(self, x: torch.Tensor):
        # get dense spectral matrix (Eq.2)
        F = torch.zeros(self.d1, self.d2)    # shape: (d1, d2)
        F[self.E[0, :], self.E[1, :]] = self.c  # 随机替换 n 个权重向量为 self.c
        # compute Delta_W (Eq.3)
        Delta_W = torch.fft.ifft2(F).real * self.alpha    # shape: (d1, d2)
        # merge (Eq.4)
        h = self.base_layer(x)    # shape: (B, L, C=d2)
        h += torch.einsum('ijk,kl->ijl', x, Delta_W)
        return h

base_linear = nn.Linear(4096, 4096)
fourierFT = FourierFT(base_layer = base_linear)
input = torch.randn(6, 196, 4096)
output = fourierFT(input)
