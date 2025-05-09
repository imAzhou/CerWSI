_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'uni'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'avg'  # identity, conv, avg
avg_dim = 1
# neck_output_dim = [768]

# classifier
classifier_type = 'multicls_linear'
input_embed_dim = 1024