_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'resnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'conv'  # identity, conv
neck_output_dim = [768]

# classifier
classifier_type = 'chief'
size_type = 'small'     # small: 768,   big: 1024,   large: 2048
dropout = True
