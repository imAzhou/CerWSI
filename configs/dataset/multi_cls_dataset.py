# dataset settings
data_root = 'data_resource/cls_valid'
classes = ['negative', 'positive']
train_bs = 64
val_bs = 32
img_input_size = 224

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_input_size, img_input_size)),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical','diagonal']),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_input_size, img_input_size)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=16,
    dataset=dict(
        data_root=data_root,
        ann_file='train.txt',
        data_prefix='',
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
        ann_file='val.txt',
        data_prefix='',
        with_label=True,
        classes=classes,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


val_evaluator = [
    dict(type='Accuracy'),
    # dict(type='AveragePrecision'),
    dict(type='BinaryTPRTNR'),
    # dict(type='SingleLabelMetric', average=None),  # output class-wise directly
    # dict(type='SingleLabelMetric', average='macro'),  # overall mean
    # dict(type='BinaryTPRTNR', class_map={0:0, 1:0, 2:1, 3:1}),
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
