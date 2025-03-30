
UNICONFIG = dict(
    backbone_output_dim = [1024],
    backbone_ckpt = 'checkpoints/uni.bin',
    frozen_backbone = False,
    use_peft = None,   # None, lora, FourierFT
    num_patches = 196,
)

SAMCONFIG = dict(
    backbone_output_dim = [768],
    backbone_ckpt = 'checkpoints/sam_vit_b_01ec64.pth',
    backbone_size_type = 'vit_b',
    frozen_backbone = True,
    use_peft = None,   # None, lora, FourierFT
    num_patches = 64*64,
)

backbone_cfgdict = {
    'uni': UNICONFIG,
    'sam': SAMCONFIG,
}