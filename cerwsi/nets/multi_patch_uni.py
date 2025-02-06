import torch
import torch.nn as nn
from timm import create_model
from timm.layers import resample_abs_pos_embed
from peft import LoraConfig, get_peft_model
import math
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from cerwsi.utils import contrastive_loss

class MultiPatchUNI(nn.Module):
    def __init__(self, num_classes, temperature):
        super(MultiPatchUNI, self).__init__()

        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.lora_config = LoraConfig(
            r=8,  # LoRA 的秩
            lora_alpha=16,  # LoRA 的缩放因子
            target_modules = ["qkv", "proj", "fc1", "fc2"],  # 应用 LoRA 的目标模块
            lora_dropout=0.1,  # Dropout 概率
            bias="none",  # 是否调整偏置
        )
        self.temperature = temperature
        
        self.embed_dim = self.backbone.embed_dim
        # self.feat_cluster_conv = nn.Conv2d(self.embed_dim, self.embed_dim, 3,1,1)
        # self.cls_linear_head = nn.Linear(self.embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, num_classes, self.embed_dim))
        num_patches = self.backbone.patch_embed.num_patches
        embed_len = num_patches + num_classes
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        self.cls_neg_head = nn.Linear(self.embed_dim + (num_classes-1), 1)
        self.cls_pos_heads = nn.ModuleList()
        for i in range(num_classes-1):
            self.cls_pos_heads.append(nn.Linear(num_patches, 1))
        
        self.drop_neg_ratio = 0.5 # 样本平衡因子，阴性token丢弃比例
        self.feat_loss_factor = 0.5
        self.token_linear_head = nn.Linear(self.embed_dim, 1)
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
        
        # update_params = ['cls_token']
        update_params = []
        if frozen:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                if name in update_params:
                    param.requires_grad = True
    
    def load_backbone_with_LoRA(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.backbone.load_state_dict(params_weight, strict=False))

        self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
    
    def load_ckpt_with_LoRA(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        print(self.load_state_dict(params_weight, strict=True))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        prev_grid_size = self.backbone.model.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=self.num_classes,
        )
        x = x.view(B, -1, C)
        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
        return x

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        prev_grid_size = self.backbone.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=self.num_classes,
        )
        x = x.view(B, -1, C)
        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        # embed_x = self.backbone._pos_embed(x)
        embed_x = self._pos_embed(x)
        x = self.backbone.patch_drop(embed_x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x,embed_x
    
    def calc_logits(self, x: torch.Tensor):
        '''
        Return:
            img_logits: (bs, 1)
            token_logits: (bs, num_tokens, 1)
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb,shallow_emb = self.forward_features(x)
        cls_neg_token = feature_emb[:,0,:]  # (bs, C)
        cls_pos_tokens = feature_emb[:,1:self.num_classes,:]  # (bs, num_cls-1, C)
        img_tokens = feature_emb[:,self.num_classes:,:]

        simi_matrix = torch.matmul(cls_pos_tokens, img_tokens.transpose(1, 2))  # (bs, num_cls-1, num_tokens)
        simi_matrix = (simi_matrix - simi_matrix.mean(-1, keepdim=True)) / (simi_matrix.std(-1, keepdim=True) + 1e-8)

        avg_pos_token = torch.mean(simi_matrix, dim=-1)
        overall_neg_token = torch.cat([cls_neg_token,avg_pos_token], dim=-1 )
        pred_img_neg_logits = self.cls_neg_head(overall_neg_token)  # (bs, 1)

        pred_img_pos_logits = []
        for i in range(self.num_classes-1):
            pred_img_pos_logits.append(self.cls_pos_heads[i](simi_matrix[:,i,:]))  # [(bs, 1),]
        pred_img_pos_logits = torch.cat(pred_img_pos_logits, dim=-1)  # (bs, num_cls-1)

        return pred_img_neg_logits, pred_img_pos_logits, img_tokens
    
    def create_feat_gt(self, logits_shape, databatch):
        bs,num_tokens,_ = logits_shape
        keep_neg_nums = int(num_tokens*self.drop_neg_ratio)
        feat_gt = torch.zeros((bs, num_tokens)).to(self.device)
        balanced_mask = torch.zeros((bs, num_tokens)).to(self.device)
        feat_hw = int(math.sqrt(num_tokens))
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    # 对于阳性图片来说，标记了类别的token记为1，其他为0的只是没标记，不代表没病变
                    feat_gt[bidx, (row*feat_hw)+col] = 1
                    balanced_mask[bidx, (row*feat_hw)+col] = 1
            if img_label == 0:
                # 对于阴性图片来说，它所有 token 对于类别0来说都是正样本
                feat_gt[bidx, :] = 0
                rand_keep = torch.randint(0, num_tokens, (keep_neg_nums,))
                balanced_mask[bidx, rand_keep] = 1

        return feat_gt,balanced_mask

    def calc_feat_loss(self, feat_logits, img_tokens, databatch):
        bs, num_tokens, _ = feat_logits.shape
        feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_per_token = criterion(feat_logits.view(bs * num_tokens), feat_gt.view(bs * num_tokens))
        loss_per_token = loss_per_token.view(bs, num_tokens)
        masked_loss = loss_per_token * balanced_mask
        # 计算所有类别的损失，按掩码取有效位置的损失并计算平均值
        final_clshead_loss = masked_loss.sum() / balanced_mask.sum().float()
        
        # img_tokens: (bs, num_tokens, C)   feat_gt/balanced_mask:(bs, num_tokens)
        indices = balanced_mask.nonzero(as_tuple=True)
        contrast_feat = img_tokens[indices]
        contrast_feat = F.normalize(contrast_feat, dim=-1)  # bs*num_tokens, c
        token_gt = feat_gt[indices]
        cont_loss = contrastive_loss(contrast_feat, token_gt, self.temperature)
        
        token_loss = cont_loss + final_clshead_loss
        return token_loss

    def calc_pos_loss(self, pos_logits, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = torch.zeros_like(pos_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            label_list = list(set([tk[-1] -1 for tk in token_labels]))  # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
            binary_matrix[i, label_list] = 1

        loss = loss_fn(pos_logits, binary_matrix)
        return loss

    def calc_pos_loss(self, pos_logits, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = torch.zeros_like(pos_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            label_list = list(set([tk[-1] -1 for tk in token_labels]))  # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
            binary_matrix[i, label_list] = 1

        loss = loss_fn(pos_logits, binary_matrix)
        return loss

    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def train_step(self, databatch, optim_wrapper:OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        # img_logits: (bs, 1)
        img_neg_logits,img_pos_logits,img_tokens = self.calc_logits(input_x.to(self.device))
        token_logits = self.token_linear_head(img_tokens) # (bs, num_tokens, 1)
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        
        img_neg_loss = F.binary_cross_entropy_with_logits(img_neg_logits, img_gt, reduction='mean')
        token_loss = self.calc_feat_loss(token_logits, img_tokens, databatch)
        img_pos_loss = self.calc_pos_loss(img_pos_logits, databatch)
        loss = img_neg_loss + img_pos_loss + token_loss
        
        # loss = (1-self.feat_loss_factor)*img_loss + self.feat_loss_factor*feat_loss
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        # img_logits,feat_logits = self.calc_logits(input_x.to(self.device))
        # feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch)
        img_neg_logits,img_pos_logits,_ = self.calc_logits(input_x.to(self.device))

        databatch['img_probs'] = torch.sigmoid(img_neg_logits).squeeze(-1)   # (bs, )
        databatch['pos_probs'] = torch.sigmoid(img_pos_logits)  # (bs, num_classes-1)
        return databatch
