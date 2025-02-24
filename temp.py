import torch
from cerwsi.nets.multilabel_baseline.convnext import MultiConvNext

model = MultiConvNext(5)
backbone_ckpt = 'checkpoints/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth'
model.load_backbone(backbone_ckpt,frozen=True)
inputs = torch.rand(1, 3, 224, 224)
logits = model.calc_logits(inputs)
print()