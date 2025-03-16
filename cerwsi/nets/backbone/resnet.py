import torch
from timm import create_model
from .meta_backbone import MetaBackbone

class ResNet(MetaBackbone):
    def __init__(self, args):
        super(ResNet, self).__init__(args)
        self.module = create_model("resnet50", pretrained=False, num_classes=0)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.module.load_state_dict(params_weight, strict=False))

    def forward(self, x: torch.Tensor):
        # feature_emb.shape: (bs, c, h, w)
        feature_emb = self.module.forward_features(x)
        return feature_emb
