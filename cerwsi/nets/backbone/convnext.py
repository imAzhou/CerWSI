import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class ConvNeXt(MetaBackbone):
    def __init__(self, **args):
        self.backbone = get_model(
            'convnext-large_in21k-pre_3rdparty_in1k', pretrained=False,
            backbone=dict(gap_before_final_norm=False)
        ).backbone
        output_embed_dim = 1536
        num_tokens = -1
        super(ConvNeXt, self).__init__(output_embed_dim, num_tokens, **args)

    def load_backbone(self, ckpt):
        state_dict = (torch.load(ckpt, map_location=self.device))['state_dict']
        new_state_dict = {}
        for key,value in state_dict.items():
            new_name = key.replace('backbone.', '')
            new_state_dict[new_name] = value
        print(self.backbone.load_state_dict(new_state_dict, strict=False))
    
    def forward(self, x: torch.Tensor):
        # feature_emb.shape: (bs, h, w, C)
        feature_emb = (self.backbone(x))[0]
        return feature_emb
