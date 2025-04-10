# dataset settings 
# data_root = '/x22201018/datasets/CervicalDatasets/LCerScanv2'
# data_root = '/disk/medical_datasets/cervix/data_resource'
data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv2'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
# classes = ['negative', 'ASC-US_LSIL', 'ASC-H_HSIL', 'AGC']
num_classes = len(classes)
train_bs = 48
val_bs = 48
split_group = 1     # 一般情况下设置为1，不等于1时，会分 split_group 组数据分别依次送入backbone抽特征后，组合到一起送入解码器中训练

train_annojson = 'train_patches_v0403.json'
val_annojson = 'val_patches_v0403.json'
