import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from mmengine.optim import OptimWrapper

class MultiPatchUNI(nn.Module):
    def __init__(self, num_classes, device, frozen_backbone=True):
        super(MultiPatchUNI, self).__init__()

        self.backbone = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        embed_dim = self.backbone.embed_dim
        self.cls_linear_head = nn.Linear(embed_dim, 1)
        self.token_linear_heads = nn.ModuleList()
        for i in range(num_classes):
            self.token_linear_heads.append(nn.Linear(embed_dim, 1))
        self.num_classes = num_classes
        self.device = device
        self.frozen_backbone = frozen_backbone

        self.load_backbone()

    def load_backbone(self):
        params_weight = torch.load("checkpoints/pytorch_model.bin", map_location="cpu")
        print(self.backbone.load_state_dict(params_weight, strict=True))
        if self.frozen_backbone:
            for name, param in self.backbone.named_parameters():
                if name != 'cls_token':
                    param.requires_grad = False

    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, 1)
            feat_logits: list(bs, num_tokens, 1) list len equal num_class
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb = self.backbone.forward_features(x)
        cls_token = feature_emb[:,0,:]  # (bs, C)
        feature_tokens = feature_emb[:,1:,:]  # (bs, nums, C)
        
        img_logits = self.cls_linear_head(cls_token)
        feat_logits = []
        for cls_head in self.token_linear_heads:
            feat_logits.append(cls_head(feature_tokens))

        return img_logits,feat_logits
    
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
