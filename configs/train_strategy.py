
# strategy
lr = 0.0001
min_lr = 0.00001
weight_decay = 0.001
max_epochs = 100
warmup_epoch = 5
gamma = 0.9
save_each_epoch = False
frozen_backbone = False

baseline_backbone = 'vit'
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay)
)
