data_root = '/disk/zly/slide_token'
classes = ['negative', 'positive']
num_classes = len(classes)
train_bs = 4
val_bs = 4

train_config = dict(
    root_dir = f'{data_root}/train',
    bs = train_bs,
    csv_filepath = 'data_resource/cls_pn/1127_train.csv'
)
val_config = dict(
    root_dir = f'{data_root}/train',
    bs = val_bs,
    csv_filepath = 'data_resource/cls_pn/1127_train.csv'
)

dataset_cfg = dict(
    train = train_config,
    val = val_config,
)

