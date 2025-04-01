import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np

# 自定义数据集类
class ClsDataset(Dataset):
    def __init__(self, root_dir, annojson_path, transform, classes):
        """
        Args:
            img_dir (str): img dir
        """

        self.img_dir = f'{root_dir}/images'
        self.annofiles_dir = f'{root_dir}/annofiles'

        with open(annojson_path, 'r') as f:
            self.patch_infolist = json.load(f)
            # self.patch_infolist = self.patch_infolist[:10000]
        
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.patch_infolist)

    def __getitem__(self, idx):
        imginfo = self.patch_infolist[idx]
        
        imgpath = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["filename"]}'
        image = Image.open(imgpath)

        input_tensor = self.transform(image)
        image_label = imginfo['diagnose']
        multi_pos_labels = self.get_mliti_pos_labels(imginfo)
        # 0为阴性，阳性id都会大于0
        gt_bboxes_clsid = [self.classes.index(name) for name in imginfo['clsnames']]
        clsid_mask = self.generate_bbox_mask(imginfo['bboxes'], gt_bboxes_clsid, image.size)

        return input_tensor,image_label,multi_pos_labels,clsid_mask,imgpath

    def get_mliti_pos_labels(self, imginfo):
        # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
        multi_pos_labels = torch.zeros((self.num_classes-1,))
        if 'gtmap_14' in imginfo:
            label_list = list(set([tk[-1]-1 for tk in imginfo['gtmap_14']]))
        else:
            label_list = list(set([i-1 for i in imginfo['clsid']]))
        multi_pos_labels[label_list] = 1
        return multi_pos_labels
    
    def generate_bbox_mask(self, bboxes, bboxes_clsid, shape):
        """
        生成一个形状为 shape 的矩阵，初始为 0，
        bbox 区域填充对应类别 ID，并优先填充小框的类别 ID。
        
        :param bboxes: List of bboxes [[x1, y1, x2, y2], ...]
        :param bboxes_clsid: List of clsid [1,3,...]
        :param shape: (height, width) 矩阵的目标尺寸
        :return: 生成的矩阵
        """
        h, w = shape
        mask = torch.zeros((h, w), dtype=torch.int32)
        
        # 按照 bbox 面积从大到小排序，确保小框后填充
        bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        
        for (x1, y1, x2, y2), class_id in zip(bboxes,bboxes_clsid):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))  # 限制边界
            mask[y1:y2, x1:x2] = class_id  # 填充类别 ID
        
        return mask