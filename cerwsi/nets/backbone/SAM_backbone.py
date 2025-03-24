import torch
from peft import LoraConfig, FourierFTConfig, get_peft_model
from functools import partial
from types import SimpleNamespace
from .SAM.image_encoder import ImageEncoderViT

from .meta_backbone import MetaBackbone

def get_peft_config(peft_type:str):
    if peft_type == 'lora':
        return LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["attn.qkv", "attn.proj", "lin1", "lin2"],  # 应用 LoRA 的目标模块
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

def get_backbone_config(backbone_type):
    configs = {
        "vit_h": dict(
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
        ),
        "vit_l": dict(
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[5, 11, 17, 23],
        ),
        "vit_b": dict(
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
        ),
    }

    return configs[backbone_type]

class SAMEncoder(MetaBackbone):
    def __init__(self, args):
        super(SAMEncoder, self).__init__(args)
        image_size = args.img_size
        vit_patch_size = 16
        out_chans = 256
        image_embedding_size = image_size // vit_patch_size
        encoder_cfg = get_backbone_config(args.backbone_size_type)
        encoder_cfg = SimpleNamespace(**encoder_cfg)
        self.backbone = ImageEncoderViT(
            depth=encoder_cfg.encoder_depth,
            embed_dim=encoder_cfg.encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_cfg.encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_cfg.encoder_global_attn_indexes,
            window_size=14,
            out_chans=out_chans,
        )
        if args.backbone_ckpt is not None:
            self.load_backbone(args.backbone_ckpt)

        if args.frozen_backbone:
            self.freeze_backbone()

        if args.use_peft is not None:
            self.peft_config = get_peft_config(args.use_peft)
            self.backbone = get_peft_model(self.backbone, self.peft_config).base_model

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location='cpu')
        state_dict = {}
        for key,value in params_weight.items():
            if 'image_encoder' in key:
                new_name = key.replace('image_encoder.', '')
                state_dict[new_name] = value
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        print('Load backbone SAM: ' + str(load_result))

    def freeze_backbone(self):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        embed_256,inter_feature = self.backbone(x, need_inter=True)
        # (-1, h=64, w=64, c=1280)
        output = inter_feature[-1].flatten(start_dim=1, end_dim=2)  # (bs, num_tokens, C)
        return output

    
    
    
    