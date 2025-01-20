import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from mmengine.optim import OptimWrapper

class MultiPatchUNI(nn.Module):
    def __init__(self, num_classes):
        super(MultiPatchUNI, self).__init__()

        self.backbone = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.cls_linear_head = nn.Linear(self.embed_dim, 1)
        self.pos_neg_ratio = 5 # 样本平衡因子，正负样本数量多的一方最多只能超出另一方5倍
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
        
        update_params = ['cls_token','cls_linear_head','token_linear_head']
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                if name in update_params:
                    param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x
    
    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, 1)
            feat_logits_list: list(bs, num_tokens, 1) list len equal num_class
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb = self.forward_features(x)
        cls_token = feature_emb[:,0,:]  # (bs, C)
        feature_tokens = feature_emb[:,1:,:]  # (bs, nums, C)
        
        img_logits = self.cls_linear_head(cls_token)  # (bs, nums, num_classes)
        # feat_logits_list = []
        # for cls_head in self.token_linear_heads:
        #     feat_logits_list.append(cls_head(feature_tokens))

        feat_logits = self.token_linear_head(feature_tokens)
        return img_logits,feat_logits
    
    def create_balanced_mask(self, flatten_feat_gt):
        """
        Create a balanced mask for sample selection across all classes.
        
        Args:
            flatten_feat_gt (torch.Tensor): Tensor of shape (num_class, sample_nums) with values 0 or 1.
        Returns:
            torch.Tensor: A mask tensor with the same shape as flatten_feat_gt.
        """
        mask = torch.zeros_like(flatten_feat_gt, dtype=torch.int32).to(self.device)
        
        # Class 0 mask creation
        class0_indices = torch.nonzero(flatten_feat_gt[0] == 1, as_tuple=False).squeeze(1)
        other_class_indices = torch.nonzero(flatten_feat_gt[1:].sum(dim=0) > 0, as_tuple=False).squeeze(1)
        
        min_class0_count = min(len(class0_indices), len(other_class_indices))
        num_class0_neg_samples = min_class0_count * self.pos_neg_ratio
        
        if len(other_class_indices) > num_class0_neg_samples:
            selected_neg_indices_class0 = other_class_indices[torch.randperm(len(other_class_indices))[:num_class0_neg_samples]]
        else:
            selected_neg_indices_class0 = other_class_indices
        
        mask[0, class0_indices] = 1
        mask[0, selected_neg_indices_class0] = 1
        
        # Other classes' mask creation
        for cls in range(1, self.num_classes):
            pos_indices = torch.nonzero(flatten_feat_gt[cls] == 1, as_tuple=False).squeeze(1)
            if len(pos_indices) == 0:
                continue
            num_neg_samples = len(pos_indices) * self.pos_neg_ratio
            neg_candidates = class0_indices
            if len(neg_candidates) > num_neg_samples:
                neg_indices = neg_candidates[torch.randperm(len(neg_candidates))[:num_neg_samples]]
            else:
                neg_indices = neg_candidates
            
            mask[cls, pos_indices] = 1
            mask[cls, neg_indices] = 1
        return mask
    
    def create_feat_gt(self, logits_shape, databatch):
        bs,num_tokens,_ = logits_shape
        feat_gt = torch.zeros((self.num_classes, bs, num_tokens)).to(self.device)
        
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    # 对于阳性图片来说，标记了类别的token记为1，其他为0的只是没标记，不代表没病变
                    feat_gt[clsid, bidx, (row*14)+col] += 1
            if img_label == 0:
                # 对于阴性图片来说，它所有 token 对于类别0来说都是正样本
                feat_gt[0, bidx, :] += 1
        return feat_gt

    def calc_feat_loss(self, feat_logits, databatch):
        feat_gt = self.create_feat_gt(feat_logits_list[0].shape, databatch)
        flatten_feat_logits = torch.stack(feat_logits_list).flatten(1) # (num_cls, bs*num_tokens)
        flatten_feat_gt = feat_gt.flatten(1) # (num_cls, bs*num_tokens)
        flatten_balanced_mask = self.create_balanced_mask(flatten_feat_gt) # (num_cls, bs*num_tokens)
        
        # 计算所有类别的损失，按掩码取有效位置的损失并计算平均值
        clshead_loss = F.binary_cross_entropy_with_logits(flatten_feat_logits, flatten_feat_gt, reduction='none')
        masked_loss = (clshead_loss * flatten_balanced_mask).sum(dim=1) / flatten_balanced_mask.sum(dim=1).clamp(min=1e-6)
        final_clshead_loss = masked_loss.mean()

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
        img_logits,feat_logits_list = self.calc_logits(input_x.to(self.device))
        databatch['feat_gt'] = self.create_feat_gt(feat_logits_list[0].shape, databatch)  # num_classes, bs, num_tokens
        databatch['img_probs'] = torch.sigmoid(img_logits).squeeze(-1)   # (bs, )
        databatch['feat_probs'] = torch.sigmoid(torch.stack(feat_logits_list)).squeeze(-1)  # (num_class, bs, num_tokens)
        return databatch
