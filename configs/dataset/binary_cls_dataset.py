# dataset settings
data_root = '/x22201018/datasets/CervicalDatasets/data_resource'
classes = ['negative', 'positive']
num_classes = len(classes)
train_bs = 128
val_bs = 128
img_input_size = 518
train_ann_file = 'annofile/1231_train_c2.txt'
val_ann_file = 'annofile/1231_val_c2.txt'
data_prefix = ''

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_input_size, img_input_size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical','diagonal']),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_input_size, img_input_size), keep_ratio=True),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=16,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=data_prefix,
        with_label=True,
        classes=classes,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=val_bs,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=data_prefix,
        with_label=True,
        classes=classes,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='BinaryTPRTNR'),
    dict(type='SingleLabelMetric', average=None),  # output class-wise directly
    dict(type='SingleLabelMetric', average='macro'),  # overall mean
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
