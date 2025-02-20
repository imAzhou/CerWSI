from timm import create_model
from mmpretrain import get_model

def get_backbone(backbone_type):
    '''
    backbone_type: ['vit', 'dinov2', 'uni']
    '''
    if backbone_type == 'uni':
        backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        feature_dim = backbone.embed_dim
        num_patches = backbone.patch_embed.num_patches
    elif backbone_type in ['vit', 'dinov2']:
        backbone_model_name = {
            'vit': 'vit-large-p16_in21k-pre_3rdparty_in1k-384px',
            'dinov2': 'vit-large-p14_dinov2-pre_3rdparty'
        }
        backbone = get_model(
            backbone_model_name[backbone_type], 
            pretrained=False, 
            backbone=dict(out_type='raw', with_cls_token=False)
        ).backbone
        feature_dim = backbone.embed_dims
        num_patches = backbone.patch_resolution[0] * backbone.patch_resolution[1]
    return backbone,feature_dim,num_patches