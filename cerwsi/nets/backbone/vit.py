import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class ViT(MetaBackbone):
    def __init__(self, **args):
        '''
        out_type: 'cls_token' / 'raw'
        with_cls_token: True / False
        '''
        backbone_type = 'vit-l'
        backbone_model_name = {
            'vit-l': 'vit-large-p16_in21k-pre_3rdparty_in1k-384px',
        }
        self.backbone = get_model(
            backbone_model_name[backbone_type], 
            pretrained=False,
            backbone=dict(out_type=args.out_type, with_cls_token=args.with_cls_token)
        ).backbone
        output_embed_dim = self.backbone.embed_dims
        patch_size = 16
        feat_size = args.img_size // patch_size
        num_tokens = feat_size * feat_size
        super(ViT, self).__init__(output_embed_dim, num_tokens, **args)

    def load_backbone(self, ckpt):
        state_dict = torch.load(ckpt, map_location=self.device)
        new_state_dict = {}
        for key,value in state_dict.items():
            new_name = key.replace('backbone.', '')
            new_state_dict[new_name] = value
        print(self.backbone.load_state_dict(new_state_dict, strict=False))
    
    def forward(self, x: torch.Tensor):
        feature_emb = (self.backbone(x))[0]
        return feature_emb
