# dataset settings 
# data_root = '/x22201018/datasets/CervicalDatasets/data_resource'
# data_root = '/disk/medical_datasets/cervix/data_resource'
data_root = '/c22073/zly/datasets/CervicalDatasets/data_resource'
classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
# classes = ['negative', 'ASC-US_LSIL', 'ASC-H_HSIL', 'AGC']
num_classes = len(classes)
train_bs = 128
val_bs = 128

img_size = 224

train_annojson = f'{data_root}/annofiles/train_patches_v0309.json'
val_annojson = f'{data_root}/annofiles/val_patches_v0309.json'
