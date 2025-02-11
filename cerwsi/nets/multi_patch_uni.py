import torch
import torch.nn as nn
from timm import create_model
from timm.layers import resample_abs_pos_embed
from peft import LoraConfig, get_peft_model
from mmengine.optim import OptimWrapper
from .classifier import CerMClassifier

class MultiPatchUNI(nn.Module):
    def __init__(self, num_classes, use_lora, temperature):
        super(MultiPatchUNI, self).__init__()

        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.use_lora = use_lora
        self.num_classes = num_classes
        if use_lora:
            self.lora_config = LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["qkv", "proj", "fc1", "fc2"],  # 应用 LoRA 的目标模块
                lora_dropout=0.1,  # Dropout 概率
                bias="none",  # 是否调整偏置
            )
            self.temperature = temperature
        
        self.embed_dim = self.backbone.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, num_classes, self.embed_dim))
        num_patches = self.backbone.patch_embed.num_patches
        embed_len = num_patches + num_classes
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        
        self.classifier = CerMClassifier(num_classes,num_patches,self.embed_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def load_backbone(self, ckpt, frozen=True):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.backbone.load_state_dict(params_weight, strict=False))
        
        if self.use_lora:
            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        else:
            # update_params = ['cls_token']
            update_params = []
            if frozen:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False
                    if name in update_params:
                        param.requires_grad = True
    
    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        if self.use_lora:
            self.backbone = get_peft_model(self.backbone, self.lora_config).base_model
        
        print(self.load_state_dict(params_weight, strict=True))


    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if self.use_lora:
            prev_grid_size = self.backbone.model.patch_embed.grid_size
        else:
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
    
    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def train_step(self, databatch, optim_wrapper:OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        # img_logits: (bs, 1)
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb,shallow_emb = self.forward_features(input_x.to(self.device))
        pred_logits = self.classifier.calc_logits(feature_emb)
        loss = self.classifier.calc_loss(pred_logits, databatch)
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        feature_emb,shallow_emb = self.forward_features(input_x.to(self.device))
        pred_logits = self.classifier.calc_logits(feature_emb)
        img_neg_logits,img_pos_logits,token_logits = pred_logits

        databatch['img_probs'] = torch.sigmoid(img_neg_logits).squeeze(-1)   # (bs, )
        databatch['pos_probs'] = torch.sigmoid(img_pos_logits)  # (bs, num_classes-1)
        return databatch
