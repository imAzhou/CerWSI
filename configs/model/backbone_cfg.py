RESNET = dict(
    backbone_output_dim = [2048],
    backbone_ckpt = 'checkpoints/resnet50_a1_0-14fe96d1.pth',
    frozen_backbone = False,
    use_peft = None,
)

CTRANSPATH = dict(
    backbone_output_dim = [768],
    backbone_ckpt = 'checkpoints/ctranspath.pth',
    frozen_backbone = False,
    use_peft = None,
)

UNICONFIG = dict(
    backbone_output_dim = [1024],
    backbone_ckpt = 'checkpoints/uni.bin',
    frozen_backbone = False,
    use_peft = None,   # None, lora, FourierFT, dtcwt
    vit_patch_size = 16
)

SAMCONFIG = dict(
    backbone_output_dim = [768],
    backbone_ckpt = 'checkpoints/sam_vit_b_01ec64.pth',
    backbone_size_type = 'vit_b',
    frozen_backbone = True,
    use_peft = 'lora', 
    use_dtcwt_indexes = [0,1],
    vit_patch_size = 16
)

backbone_cfgdict = {
    'resnet': RESNET,
    'uni': UNICONFIG,
    'sam': SAMCONFIG,
}