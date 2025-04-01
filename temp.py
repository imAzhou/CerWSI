from cerwsi.nets.backbone.SVT_backbone import SVTBackbone
import torch

model = SVTBackbone(None)
# with open('model_structure/SVT.txt', 'w') as f:
#     f.writelines(str(model))
bs,c,h,w = 6,3,224,224
input = torch.randn((bs,c,h,w))
output = model(input)
print()