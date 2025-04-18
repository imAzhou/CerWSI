import torch
import torch.nn as nn
from .meta_backbone import MetaBackbone
from .CTransPath import SwinTransformer

class CTransPath(MetaBackbone):
    def __init__(self, args):
        super(CTransPath, self).__init__()
        self.backbone = SwinTransformer()
        output_embed_dim = 768
        num_tokens = 7*7
        super(CTransPath, self).__init__(args)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.backbone.load_state_dict(params_weight['model'], strict=False))

    def forward(self, x: torch.Tensor):
        # feature_emb:  (B L C)
        feature_emb = self.backbone.forward_features(x)
        return feature_emb
