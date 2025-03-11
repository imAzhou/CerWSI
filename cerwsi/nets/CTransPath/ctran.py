import torch.nn as nn
import torch
from .swin_transformer import SwinTransformer
from .conv_stem import ConvStem

def ctranspath(ckpt = None):
    model = SwinTransformer(embed_layer = ConvStem)
    model.head = nn.Identity()
    if ckpt:
        td = torch.load(ckpt)
        model.load_state_dict(td['model'], strict=True)
    return model