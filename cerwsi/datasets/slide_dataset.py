import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import glob

# 自定义数据集类
class SlideDataset(Dataset):
    def __init__(self, root_dir, transform):

        self.total_imgs = glob.glob(f'{root_dir}/patientImgs/**/*.png')      
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        imgpath = self.total_imgs[idx]
        patientId = os.path.dirname(imgpath).split('/')[-1]
        image = Image.open(imgpath)
        input_tensor = self.transform(image)
        return input_tensor,patientId,imgpath
