import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import json
from torchvision import transforms

# 自定义数据集类
class TokenClsDataset(Dataset):
    def __init__(self, root_dir, mode):
        """
        Args:
            img_dir (str): img dir
        """

        self.img_dir = f'{root_dir}/images'
        self.annofiles_dir = f'{root_dir}/annofiles'

        with open(f'{self.annofiles_dir}/{mode}_patches.json', 'r') as f:
            self.patch_infolist = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.patch_infolist)

    def __getitem__(self, idx):
        imginfo = self.patch_infolist[idx]
        
        imgpath = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["filename"]}'
        image = Image.open(imgpath)

        input_tensor = self.transform(image)
        image_label = imginfo['diagnose']
        token_label = imginfo['gtmap_14']

        return input_tensor,image_label,token_label

