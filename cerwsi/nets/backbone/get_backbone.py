from .resnet import ResNet
from .convnext import ConvNeXt
from .vit import ViT
from .dinov2 import DINOV2
from .uni import UNI
from .ctranspath import CTransPath
from .SVT_backbone import SVTBackbone
from .SAM_backbone import SAMEncoder

allowed_backbone_type = ['resnet', 'convnext', 'vit', 
                         'dinov2', 'uni', 'ctranspath', 'svt', 'sam']

def get_backbone(args):
    backbone_type = args.backbone_type
    assert backbone_type in allowed_backbone_type, f'backbone_type allowed in {allowed_backbone_type}'
    
    backbone = None
    if backbone_type == 'resnet':
        backbone = ResNet
    if backbone_type == 'convnext':
        backbone = ConvNeXt
    if backbone_type == 'vit':
        backbone = ViT
    if backbone_type == 'dinov2':
        backbone = DINOV2
    if backbone_type == 'uni':
        backbone = UNI
    if backbone_type == 'ctranspath':
        backbone = CTransPath
    if backbone_type == 'svt':
        backbone = SVTBackbone
    if backbone_type == 'sam':
        backbone = SAMEncoder
    
    return backbone(args)