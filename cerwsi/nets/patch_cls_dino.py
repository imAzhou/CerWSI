import torch
import torch.nn as nn
from dinov2.hub.backbones import dinov2_vits14
import torch.nn.functional as F
from mmpretrain.models import ClsDataPreprocessor
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper

class PatchClsDINO(nn.Module):
    def __init__(self, num_classes, device, arch_name='vit_small', layers=4):
        super(PatchClsDINO, self).__init__()

        self.backbone = dinov2_vits14(pretrained=False)
        embed_dim = self.backbone.embed_dim
        self.linear_head = nn.Linear((1 + layers) * embed_dim, num_classes)
        self.to(device)
        
        self.data_preprocessor = ClsDataPreprocessor(
            num_classes=num_classes,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )
        self.layers = layers
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def load_backbone(self, ckpt_path, frozen=True):
        backbone_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        print(self.backbone.load_state_dict(backbone_ckpt, strict=False))
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)

    def train_step(self, data_batch, optim_wrapper:OptimWrapper):
        data = self.data_preprocessor(data_batch, True)
        logits = self.forward(data['inputs'].to(self.device))
        gt_label = torch.as_tensor([item.gt_label for item in data['data_samples']]).to(self.device)
        loss = self.loss_fn(logits, gt_label)

        optim_wrapper.update_params(loss)

        return loss

    def val_step(self, data_batch):
        data = self.data_preprocessor(data_batch, False)
        cls_score = self.forward(data['inputs'].to(self.device))
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        data_samples = data['data_samples']
        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples


if __name__ == '__main__':
    net = PatchClsDINO(num_classes=6)
