# dataset settings 

data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv3'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
train_bs = 8
val_bs = 8
split_group = 1     # 一般情况下设置为1，不等于1时，会分 split_group 组数据分别依次送入backbone抽特征后，组合到一起送入解码器中训练

train_annojson = 'train_v0410.json'
val_annojson = 'val_v0410.json'
