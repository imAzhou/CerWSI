
# strategy
lr = 0.001
weight_decay = 0.001
max_epochs = 50
warmup_epoch = 5
gamma = 0.9
save_each_epoch = False

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay)
)