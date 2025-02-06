# dataset settings 
# data_root = '/nfs5/zly/codes/CerWSI/data_resource/0103'
data_root = '/disk/medical_datasets/cervix/data_resource'
# classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
classes = ['negative', 'ASC-US_LSIL', 'ASC-H_HSIL', 'AGC']
num_classes = len(classes)
train_bs = 96
val_bs = 96
