import random
import numpy as np
import torch
import chardet
import json
from typing import Any, Generator, List

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_M = 1e6
    return {
        'Total': f'{(total_num/one_M):.4f}M',
        'Trainable': f'{(trainable_num/one_M):.4f}M',
    }

def is_bbox_inside(bbox1, bbox2):
    """
    判断 bbox1 是否被包含在 bbox2 内部
    bbox 格式为 [x_min, y_min, x_max, y_max]
    """
    return (bbox1[0] >= bbox2[0] and  # bbox1的左边界在bbox2的左边界之内
            bbox1[1] >= bbox2[1] and  # bbox1的上边界在bbox2的上边界之内
            bbox1[2] <= bbox2[2] and  # bbox1的右边界在bbox2的右边界之内
            bbox1[3] <= bbox2[3])     # bbox1的下边界在bbox2的下边界之内

def random_cut_fn(x1,y1,w,h, cut_num=1):
    if w < 64 or h < 64:
        maxlen = int(max(w,h) * random.uniform(1.5, 4))
        interval = sorted([128, maxlen])
    elif w > 224 and h > 224:
        maxlen = int(max(w,h) * random.uniform(1.1, 1.5))
        interval = [max(w,h), maxlen]
    else:
        maxlen = int(max(w,h) * random.uniform(1.5, 2))
        interval = [max(w,h), maxlen]
    
    cut_results = []
    for _ in range(cut_num):
        new_w,new_h = random.randint(interval[0],interval[1]),random.randint(interval[0],interval[1])
        minx,miny = x1-(new_w-w), y1-(new_h-h)
        maxx,maxy = x1, y1
        newx,newy = random.randint(minx,maxx), random.randint(miny,maxy)
        assert is_bbox_inside([x1,y1,x1+w,y1+h], [newx,newy,newx+new_w,newy+new_h]), "new box cannot contained the original box"
        cut_results.append([newx,newy,new_w,new_h])
    return cut_results

def read_json_anno(json_path):
    with open(json_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open(json_path,'r',encoding = encoding) as f:
        data = json.load(f)
    annotations = data['annotation']

    return annotations

def read_json_valid(json_path, max_xy):
    '''
    返回江丰生物json标注中的有效标注框

    Args:
        json_path: str
        max_xy: (max_x, max_y)
    Return:
        valid_annos: list(dict(coord=(x1,y1,x2,y2), size=(w,h), patch_clsname=''))
    '''
    NEGATIVE_CLASS = ['NILM', 'GEC']
    POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']

    with open(json_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open(json_path,'r',encoding = encoding) as f:
        data = json.load(f)
    annotations = data['annotation']
    max_x, max_y = max_xy

    pos_bbox_record, neg_bbox_record = [], []
    for idx,i in enumerate(annotations):
        region = i.get('region')
        sub_class = i.get('sub_class')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        if x2 > max_x or y2 > max_y:
            continue
        if sub_class in NEGATIVE_CLASS and (w>224 and h>224):
            neg_bbox_record.append((idx, [x1,y1,x2,y2], sub_class))
        if sub_class in POSITIVE_CLASS and (w>100 and h>100):
            pos_bbox_record.append((idx, [x1,y1,x2,y2], sub_class))

    valid_annos = []
    for bbox_info in neg_bbox_record:
        valid_flag = True
        for idx,i in enumerate(annotations):
            region = i.get('region')
            sub_class = i.get('sub_class')
            w,h = region['width'],region['height']
            x1,y1 = region['x'],region['y']
            x2,y2 = x1+w, y1+h
            if idx != bbox_info[0] and is_bbox_inside([x1,y1,x2,y2], bbox_info[1]) and sub_class not in NEGATIVE_CLASS:
                valid_flag = False
                break
        if valid_flag:
            bx1,by1,bx2,by2 = bbox_info[1]
            bw,bh = bx2 - bx1, by2 - by1
            valid_annos.append(dict(coord=(bx1,by1,bx2,by2), size=(bw,bh), patch_clsname=bbox_info[2]))
    
    for bbox_info in pos_bbox_record:
        valid_flag = True
        for idx,i in enumerate(annotations):
            region = i.get('region')
            sub_class = i.get('sub_class')
            w,h = region['width'],region['height']
            x1,y1 = region['x'],region['y']
            x2,y2 = x1+w, y1+h
            if idx != bbox_info[0] and is_bbox_inside([x1,y1,x2,y2], bbox_info[1]) and sub_class != bbox_info[2]:
                valid_flag = False
                break
        if valid_flag:
            bx1,by1,bx2,by2 = bbox_info[1]
            bw,bh = bx2 - bx1, by2 - by1
            valid_annos.append(dict(coord=(bx1,by1,bx2,by2), size=(bw,bh), patch_clsname=bbox_info[2]))
    
    return valid_annos