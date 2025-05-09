import torch
from peft import LoraConfig, FourierFTConfig, get_peft_model
from timm import create_model
from timm.layers import resample_abs_pos_embed
from .meta_backbone import MetaBackbone

def get_peft_config(peft_type:str):
    if peft_type == 'lora':
        return LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["qkv", "proj", "fc1", "fc2"],  # 应用 LoRA 的目标模块
                lora_dropout=0.1,  # Dropout 概率
                bias="none",  # 是否调整偏置
            )
    if peft_type == 'FourierFT':
        return FourierFTConfig(
            n_frequency = 1000,
            target_modules = ["qkv", "proj", "fc1", "fc2"],
            exclude_modules = ["patch_embed.proj"],
            scaling = 300.0
        )

class UNI(MetaBackbone):
    def __init__(self, args):
        super(UNI, self).__init__(args)
        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        use_peft = args.backbone_cfg['use_peft']

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

        if use_peft is not None:
            self.peft_config = get_peft_config(use_peft)
            self.backbone = get_peft_model(self.backbone, self.peft_config).base_model

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location='cpu')
        load_result = self.backbone.load_state_dict(params_weight, strict=False)
        print('Load backbone NUI: ' + str(load_result))

    def forward(self, x: torch.Tensor):
        output = self.backbone.forward_features(x)
        output = output[:,1:,:]  # (bs, num_tokens, C)
        return output.transpose(1,2)
    
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
            token_logits: (bs, num_tokens, 1)
        '''
        # feature_emb.shape: (bs,cls_token+img_token, C)
        feature_emb = self.forward_features(x)
        cls_token = feature_emb[:,0,:]  # (bs, C)

        pred_img_logits = []
        for i in range(self.num_classes-1):
            pred_img_logits.append(self.cls_linear_heads[i](cls_token))  # [(bs, 1),]
        
        pred_img_logits = torch.cat(pred_img_logits, dim=-1)  # (bs, num_cls)
        return pred_img_logits

    