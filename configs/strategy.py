
# strategy
lr = 0.0005
min_lr = 0.00001
weight_decay = 0.001
max_epochs = 100
warmup_epoch = 5
gamma = 0.95
save_each_epoch = True
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay),
    # clip_grad=dict(max_norm=1.0, norm_type=2)  # 添加梯度裁剪
)

positive_thr = 0.5
img_size = 512  # 224, 448, 512, 1024

logger_name = 'wscer_partial'
