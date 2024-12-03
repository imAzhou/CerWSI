# dataset settings
data_root = '/c22073/zly/datasets/WXL_JFSW'
classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
num_classes = len(classes)
train_bs = 64
val_bs = 32
img_input_size = 518
txt_tail = 'rcp_c6'    # rcp/origin
data_prefix = 'random_cut'    # random_cut/original


train_pipeline = [
    dict(type='LoadImageFromFile'),
    # RandomCrop: pad 是图片在中间
    dict(type='RandomCrop', crop_size=500, pad_if_needed=True, pad_val=(255,255,255)),
    dict(type='Resize', scale=(img_input_size, img_input_size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical','diagonal']),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=500, pad_if_needed=True, pad_val=(255,255,255)),
    dict(type='Resize', scale=(img_input_size, img_input_size), keep_ratio=True),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    num_workers=16,
    dataset=dict(
        data_root=data_root,
        ann_file=f'train_{txt_tail}.txt',
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
        ann_file=f'val_{txt_tail}.txt',
        data_prefix=data_prefix,
        with_label=True,
        classes=classes,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric', average=None),  # output class-wise directly
    dict(type='SingleLabelMetric', average='macro'),  # overall mean
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
