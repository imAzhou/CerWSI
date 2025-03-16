from mmpretrain.models.necks import GlobalAveragePooling
import torch.nn as nn

allowed_neck_type = ['avg', 'identity']

def get_neck(args):
    neck_type = args.neck_type
    assert neck_type in allowed_neck_type, f'neck_type allowed in {allowed_neck_type}'
    
    neck = None
    if neck_type == 'avg':
        neck = GlobalAveragePooling()
    if neck_type == 'identity':
        neck = nn.Identity()
    
    return neck