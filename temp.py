import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# 颜色映射
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
CLS_COLORS = {
    'ASC-US': '#ff9999',  # 浅红
    'LSIL': '#ff6666',  # 中浅红
    'ASC-H': '#ff3333',  # 中红
    'HSIL': '#cc0000',  # 深红
    'AGC': '#1f77b4'   # 蓝色
}

# 获取 token 相关数据
token_probs, token_classes = outputs['token_probs'][bidx], outputs['token_classes'][bidx]
feat_size = int(math.sqrt(token_probs.shape[0]))
metainfo = outputs['metainfo'][bidx]

# 读取图像和标注信息
img_path, bboxes_coords, bboxes_clsname = metainfo['img_path'], metainfo['bboxes'], metainfo['clsnames']
img = Image.open(img_path)
w, h = img.size  # 获取原图尺寸
filename = os.path.basename(img_path)

# 处理 token_classes 和 token_probs
token_classes = token_classes.reshape((feat_size, feat_size))
token_probs = token_probs.reshape((feat_size, feat_size))

# 最近邻插值到原图大小（token_classes）
token_classes_resized = cv2.resize(token_classes, (w, h), interpolation=cv2.INTER_NEAREST)

# 双线性插值到原图大小（token_probs）
token_probs_resized = cv2.resize(token_probs, (w, h), interpolation=cv2.INTER_LINEAR)

