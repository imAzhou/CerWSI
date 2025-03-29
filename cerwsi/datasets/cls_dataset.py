import torch
from PIL import Image
from torch.utils.data import Dataset
import json

# 自定义数据集类
class ClsDataset(Dataset):
    def __init__(self, root_dir, annojson_path, transform, num_classes):
        """
        Args:
            img_dir (str): img dir
        """

        self.img_dir = f'{root_dir}/images'
        self.annofiles_dir = f'{root_dir}/annofiles'

        with open(annojson_path, 'r') as f:
            self.patch_infolist = json.load(f)
        
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.patch_infolist)

    def __getitem__(self, idx):
        imginfo = self.patch_infolist[idx]
        
        imgpath = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["filename"]}'
        image = Image.open(imgpath)

        input_tensor = self.transform(image)
        image_label = imginfo['diagnose']
        multi_pos_labels = self.get_mliti_pos_labels(imginfo)

        return input_tensor,image_label,multi_pos_labels,imgpath

    def get_mliti_pos_labels(self, imginfo):
        # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
        multi_pos_labels = torch.zeros((self.num_classes-1,))
        if 'gtmap_14' in imginfo:
            label_list = list(set([tk[-1]-1 for tk in imginfo['gtmap_14']]))
        else:
            label_list = list(set([i-1 for i in imginfo['clsid']]))
        multi_pos_labels[label_list] = 1
        return multi_pos_labels
    