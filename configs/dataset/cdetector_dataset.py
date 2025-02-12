# dataset settings 
# data_root = '/disk/medical_datasets/cervix/ComparisonDetectorDataset'
# data_root = '/x22201018/datasets/CervicalDatasets/ComparisonDetectorDataset'
data_root = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
# classes = ['negative', 'ASC-US_LSIL', 'ASC-H_HSIL', 'AGC']
num_classes = len(classes)
train_bs = 128
val_bs = 128
