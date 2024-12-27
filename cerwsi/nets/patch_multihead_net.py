import torch
import torch.nn as nn
from dinov2.hub.backbones import dinov2_vits14
import torch.nn.functional as F
from mmpretrain.models import ClsDataPreprocessor
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper

class PatchMultiHeadNet(nn.Module):
    def __init__(self, num_classes, device, arch_name='vit_small', layers=4):
        super(PatchMultiHeadNet, self).__init__()

        self.backbone = dinov2_vits14(pretrained=False)
        embed_dim = self.backbone.embed_dim
        self.linear_heads = nn.ModuleList()
        for i in range(num_classes):
            self.linear_heads.append(
                nn.Linear((1 + layers) * embed_dim, 1)
            )

        self.data_preprocessor = ClsDataPreprocessor(
            num_classes=num_classes,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        ).to(device)
        
        self.num_classes = num_classes
        self.layers = layers
        self.device = device
        # self.loss_fn = nn.BCEWithLogitsLoss()

    def load_backbone(self, ckpt_path, frozen=True):
        backbone_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        print(self.backbone.load_state_dict(backbone_ckpt, strict=False))
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, data_batch, mode, optim_wrapper=None):
        data = self.data_preprocessor(data_batch, mode == 'train')
        
        if mode == 'train':
            return self.train_step(data, optim_wrapper)
        if mode == 'val':
            return self.val_step(data)

    def calc_logits(self, x):
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
        
        logits = [linear_head(linear_input) for linear_head in self.linear_heads]
        return logits

    def map_gt(self, datasamples):
        bs = len(datasamples)
        GT = torch.zeros(self.num_classes, bs).to(self.device)
        for class_id in range(self.num_classes):
            for sample_id, sample in enumerate(datasamples):
                labels = sample.gt_label
                if class_id == 0:  # 特殊处理类别 ID 为 0 的情况
                    GT[class_id, sample_id] = 1 if 0 in labels else 0
                else:  # 处理类别 ID 为非 0 的情况
                    if class_id in labels:
                        GT[class_id, sample_id] = 1
                    elif 0 not in labels:
                        GT[class_id, sample_id] = -1
        return GT
    
    def calc_mask_loss(self, input,target):
        # 逐元素计算损失
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # 构建掩码，排除 target == -1 的位置
        mask = (target != -1)
        # 仅对有效位置的损失求均值
        filtered_loss = loss[mask].mean()
        return filtered_loss


    def train_step(self, data, optim_wrapper:OptimWrapper):
        logits = self.calc_logits(data['inputs'].to(self.device))
        # logits: (num_class, bs)
        logits = torch.stack(logits).squeeze(-1)
        # gt_labels4each_cls: (num_class, bs)
        gt_labels4each_cls = self.map_gt(data['data_samples'])
        loss = self.calc_mask_loss(logits, gt_labels4each_cls)
        optim_wrapper.update_params(loss)

        return loss

    def val_step(self, data):
        logits = self.calc_logits(data['inputs'].to(self.device))
        # logits: (num_class, bs)
        logits = torch.stack(logits).squeeze(-1)
        # probs: (bs, num_class)
        probs = torch.sigmoid(logits).transpose(0,1)
        
        out_data_samples = []
        for data_sample, score in zip(data['data_samples'], probs):
            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples


if __name__ == '__main__':
    net = PatchMultiHeadNet(num_classes=6)
