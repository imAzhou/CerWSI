
UNICONFIG = dict(
    backbone_output_dim = [1024],
    backbone_ckpt = 'checkpoints/uni.bin',
    frozen_backbone = False,
    use_peft = None,   # None, lora, FourierFT
)

SAMCONFIG = dict(
    backbone_output_dim = [768],
    backbone_ckpt = 'checkpoints/sam_vit_b_01ec64.pth',
    backbone_size_type = 'vit_b',
    frozen_backbone = True,
    use_peft = 'lora',   # None, lora, FourierFT
)

backbone_cfgdict = {
    'uni': UNICONFIG,
    'sam': SAMCONFIG,
}