
# strategy
lr = 0.0001
min_lr = 0.000001
weight_decay = 0.001
max_epochs = 50
warmup_epoch = 5
gamma = 0.9
save_each_epoch = False
frozen_backbone = False
temperature = 0.1  # 较低值会导致对比损失中的相似度差异更为明显，从而加速模型的收敛，但也可能导致梯度爆炸

baseline_backbone = 'dinov2'
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay)
)
