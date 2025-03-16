import torch
import torch.nn as nn
from abc import abstractmethod

class MetaBackbone(nn.Module):
    def __init__(self, args):
        super(MetaBackbone, self).__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def load_backbone(self, ckpt):
        '''
        Args:
            ckpt (str): backbone checkpoint path
        '''

    @abstractmethod
    def forward(self, x: torch.Tensor):
        '''
        Args:
            x (tensor): input image tensor, shape is (bs, 3, H, W)
        Return:
            featuremap: list(tensor) or tensor, contain CLS token or not.
        '''
