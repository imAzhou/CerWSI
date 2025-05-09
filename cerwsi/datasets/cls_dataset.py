from PIL import Image
from torch.utils.data import Dataset
import json

# 自定义数据集类
class ClsDataset(Dataset):
    def __init__(self, root_dir, annojson_path, transform, classes):
        """
        Args:
            img_dir (str): img dir
        """
        self.img_dir = f'{root_dir}/images'
        self.root_dir = root_dir
        self.annofiles_dir = f'{root_dir}/annofiles'
        with open(f'{self.annofiles_dir}/{annojson_path}', 'r') as f:
            self.patch_infolist = json.load(f)
            # self.patch_infolist = self.patch_infolist[:10000]
        
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.patch_infolist)

    def __getitem__(self, idx):
        imginfo = self.patch_infolist[idx]
        imginfo['clsids'] = [self.classes.index(clsname) for clsname in imginfo['clsnames']]
        
        imgpath = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["filename"]}'
        imginfo['imgpath'] = imgpath
        image = Image.open(imgpath)
        imginfo['origin_size'] = image.size
        input_tensor = self.transform(image)
        image_label = 0 if imginfo['diagnose'] == 0 else max(imginfo['clsids'])
        # image_label = imginfo['diagnose']
        return input_tensor,image_label,imginfo
