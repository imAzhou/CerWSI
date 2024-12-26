data_root = 'data_resource/slide_token'
classes = ['negative', 'positive']
num_classes = len(classes)
train_bs = 16
val_bs = 16

train_config = dict(
    root_dir = f'{data_root}/train',
    bs = train_bs,
    csv_filepath = 'data_resource/cls_pn/1127_train.csv'
)
val_config = dict(
    root_dir = f'{data_root}/val',
    bs = val_bs,
    csv_filepath = 'data_resource/cls_pn/1127_val.csv'
)

dataset_cfg = dict(
    train = train_config,
    val = val_config,
)

