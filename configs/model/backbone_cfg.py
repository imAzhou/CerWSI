RESNET = dict(
    backbone_output_dim = [2048],
    backbone_ckpt = 'checkpoints/resnet50_a1_0-14fe96d1.pth',
    frozen_backbone = False,
    use_peft = None,
)

UNICONFIG = dict(
    backbone_output_dim = [1024],
    backbone_ckpt = 'checkpoints/uni.bin',
    frozen_backbone = False,
    use_peft = 'dtcwt',   # None, lora, FourierFT, dtcwt
)

SAMCONFIG = dict(
    backbone_output_dim = [768],
    backbone_ckpt = 'checkpoints/sam_vit_b_01ec64.pth',
    backbone_size_type = 'vit_b',
    frozen_backbone = True,
    use_peft = 'dtcwt',   # None, lora, FourierFT, dtcwt
)

backbone_cfgdict = {
    'resnet': RESNET,
    'uni': UNICONFIG,
    'sam': SAMCONFIG,
}