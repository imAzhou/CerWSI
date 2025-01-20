import torch
import torch.nn as nn
from timm import create_model
import math
import torch.nn.functional as F
from mmengine.optim import OptimWrapper

class MultiPatchUNI(nn.Module):
    def __init__(self, num_classes):
        super(MultiPatchUNI, self).__init__()

        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        
        self.embed_dim = self.backbone.embed_dim
        self.feat_cluster_conv = nn.Conv2d(self.embed_dim, self.embed_dim, 3,1,1)
        self.cls_linear_head = nn.Linear(self.embed_dim, 1)
        self.drop_neg_ratio = 0.5 # 样本平衡因子，阴性token丢弃比例
        self.feat_loss_factor = 0.5
        self.token_linear_head = nn.Linear(self.embed_dim, num_classes)
        # self.token_linear_heads = nn.ModuleList()
        # for i in range(num_classes):
        #     self.token_linear_heads.append(nn.Linear(self.embed_dim, 1))
        self.num_classes = num_classes

    @property
    def device(self):
        return next(self.parameters()).device

    def load_backbone(self, ckpt, frozen=True):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.backbone.load_state_dict(params_weight, strict=False))
        
        update_params = ['cls_token']
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                if name in update_params:
                    param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        embed_x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(embed_x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x,embed_x
    
    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, 1)
            feat_logits_list: list(bs, num_tokens, 1) list len equal num_class
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb,shallow_emb = self.forward_features(x)
        cls_token = feature_emb[:,0,:]  # (bs, C)
        img_logits = self.cls_linear_head(cls_token)

        # feature_tokens = feature_emb[:,1:,:]  # (bs, nums, C)
        feature_tokens = shallow_emb[:,1:,:]  # (bs, nums, C)
        bs,num_tokens,c = feature_tokens.shape
        feat_hw = int(math.sqrt(num_tokens))
        feat_token_2D = feature_tokens.reshape((bs,feat_hw,feat_hw,c)).permute(0,3,1,2)
        feature_tokens_cluster = self.feat_cluster_conv(feat_token_2D)
        flatten_cluster = feature_tokens_cluster.flatten(2).permute(0,2,1)
        token_fused = feature_tokens + flatten_cluster

        feat_logits = self.token_linear_head(token_fused)
        return img_logits,feat_logits
    
    def create_feat_gt(self, logits_shape, databatch):
        bs,num_tokens,_ = logits_shape
        keep_neg_nums = int(num_tokens*self.drop_neg_ratio)
        feat_gt = torch.zeros((bs, num_tokens)).long().to(self.device)
        balanced_mask = torch.zeros((bs, num_tokens)).to(self.device)
        feat_hw = int(math.sqrt(num_tokens))
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    # 对于阳性图片来说，标记了类别的token记为1，其他为0的只是没标记，不代表没病变
                    feat_gt[bidx, (row*feat_hw)+col] = clsid
                    balanced_mask[bidx, (row*feat_hw)+col] = 1
            if img_label == 0:
                # 对于阴性图片来说，它所有 token 对于类别0来说都是正样本
                feat_gt[bidx, :] = 0
                rand_keep = torch.randint(0, num_tokens, (keep_neg_nums,))
                balanced_mask[bidx, rand_keep] = 1

        return feat_gt,balanced_mask

    def calc_feat_loss(self, feat_logits, databatch):
        bs, num_tokens, num_classes = feat_logits.shape
        feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_per_token = criterion(feat_logits.view(bs * num_tokens, num_classes), feat_gt.view(bs * num_tokens))
        loss_per_token = loss_per_token.view(bs, num_tokens)
        masked_loss = loss_per_token * balanced_mask
        
        # 计算所有类别的损失，按掩码取有效位置的损失并计算平均值
        final_clshead_loss = masked_loss.sum() / balanced_mask.sum().float()
        return final_clshead_loss

    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def train_step(self, databatch, optim_wrapper:OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        # img_logits: (bs, 1)
        img_logits,feat_logits = self.calc_logits(input_x.to(self.device))
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        img_loss = F.binary_cross_entropy_with_logits(img_logits, img_gt, reduction='mean')
        feat_loss = self.calc_feat_loss(feat_logits, databatch)
        
        loss = (1-self.feat_loss_factor)*img_loss + self.feat_loss_factor*feat_loss
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        img_logits,feat_logits = self.calc_logits(input_x.to(self.device))
        feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch)
        databatch['feat_gt'] = feat_gt  # bs, num_tokens
        databatch['img_probs'] = torch.sigmoid(img_logits).squeeze(-1)   # (bs, )
        databatch['feat_probs'] = feat_logits  # (bs, num_tokens, num_classes)
        return databatch
