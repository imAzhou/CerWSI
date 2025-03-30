
# strategy
lr = 0.0001
min_lr = 0.00001
weight_decay = 0.001
max_epochs = 100
warmup_epoch = 5
gamma = 0.95
save_each_epoch = False
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay)
)

positive_thr = 0.5
img_size = 1024  # 224, 448, 512, 1024
