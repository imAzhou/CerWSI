import torch
import torch.nn as nn
from dinov2.hub.backbones import dinov2_vits14
import torch.nn.functional as F
from mmpretrain.models import ClsDataPreprocessor
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper

class PatchMultiClsNet(nn.Module):
    def __init__(self, num_classes, device, arch_name='vit_small', layers=4):
        super(PatchMultiClsNet, self).__init__()

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
        
        self.num_classes = num_classes
        self.layers = layers
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()

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
        binary_matrix = torch.zeros_like(logits, dtype=torch.float32)
        for i, datasample in enumerate(data['data_samples']):
            label_list = datasample.gt_label
            binary_matrix[i, label_list] = 1

        loss = self.loss_fn(logits, binary_matrix)
        optim_wrapper.update_params(loss)

        return loss

    def val_step(self, data_batch):
        data = self.data_preprocessor(data_batch, False)
        cls_score = self.forward(data['inputs'].to(self.device))
        probs = torch.sigmoid(cls_score)
        # threshold = 0.5  # 阈值可以调整
        # predictions = (probs >= threshold).int().detach()
        # predicted_labels = [torch.where(pred == 1)[0].tolist() for pred in predictions]
        
        # pred_scores = F.softmax(cls_score, dim=1)
        # pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        data_samples = data['data_samples']
        for data_sample, score in zip(data_samples, probs):
            if data_sample is None:
                data_sample = DataSample(num_classes=self.num_classes)
            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples


if __name__ == '__main__':
    net = PatchMultiClsNet(num_classes=6)
