_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'uni'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'identity'

# classifier
classifier_type = 'chief'
size_type = 'big'
dropout = True
