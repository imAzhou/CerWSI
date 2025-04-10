_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'sam'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'conv'
neck_output_dim = [512]

# classifier
classifier_type = 'wscer_alltoken'
