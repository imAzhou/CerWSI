# dataset settings
data_root = 'data_resource/ROI'
classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
num_classes = len(classes)
train_bs = 160
val_bs = 160
img_input_size = 518


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
        ann_file='annofile/1223_train_ann.json',
        data_prefix='',
        classes=classes,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=val_bs,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        ann_file='annofile/1223_val_ann.json',
        data_prefix='',
        classes=classes,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='MultiLabelMetric', thr=0.5, average=None),  # output class-wise directly
    dict(type='MultiLabelMetric', thr=0.5, average='macro'),  # overall mean
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
