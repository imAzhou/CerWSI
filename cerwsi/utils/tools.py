import random
import numpy as np
import torch
import json
from typing import Any, Generator, List

def set_seed(seed):
    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_M = 1e6
    return {
        'Total': f'{(total_num/one_M):.4f}M',
        'Trainable': f'{(trainable_num/one_M):.4f}M',
    }

def is_bbox_inside(bbox1, bbox2, tolerance=0):
    """
    判断 bbox1 是否被包含在 bbox2 内部（允许一定误差）
    bbox 格式为 [x_min, y_min, x_max, y_max]
    
    参数:
        bbox1: list, 表示被检测的边界框
        bbox2: list, 表示容器边界框
        tolerance: float, 允许的超出误差值
    返回:
        bool: 如果 bbox1 在 bbox2 内（允许误差）则返回 True，否则返回 False
    """
    return (bbox1[0] >= bbox2[0] - tolerance and  # bbox1的左边界可以稍微超出bbox2的左边界
            bbox1[1] >= bbox2[1] - tolerance and  # bbox1的上边界可以稍微超出bbox2的上边界
            bbox1[2] <= bbox2[2] + tolerance and  # bbox1的右边界可以稍微超出bbox2的右边界
            bbox1[3] <= bbox2[3] + tolerance)     # bbox1的下边界可以稍微超出bbox2的下边界

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

def random_cut_square(rect, sq_size):
    """
    根据输入矩形 rect (x, y, w, h) 的宽高条件，返回裁剪正方形区域的左上角坐标 (x1, y1)。
    sq_size：正方形边长
    规则：假设 sq_size = 500
    1. 如果 w > 500 且 h > 500，在矩形内部随机裁剪一块宽高为 500 的区域。
    2. 如果 w < 500 且 h < 500，随机生成一个宽高为 500 的矩形包裹住输入矩形。
    3. 如果宽或高小于 500，则生成一个宽高为 500 的矩形，包裹住短边，长边上随机。

    返回:
        tuple: 裁剪矩形区域的左上角坐标 (x1, y1)
    """
    int_rect = [round(i) for i in rect]
    x, y, w, h = int_rect
    # Case 1: Both width and height > sq_size
    if w > sq_size and h > sq_size:
        x1 = random.randint(x, x + w - sq_size)  # 随机选取左上角 x 坐标
        y1 = random.randint(y, y + h - sq_size)  # 随机选取左上角 y 坐标
        return x1, y1

    # Case 2: Both width and height < sq_size
    elif w < sq_size and h < sq_size:
        x1 = random.randint(x - sq_size + w, x)  # 左上角的 x1 随机
        y1 = random.randint(y - sq_size + h, y)  # 左上角的 y1 随机
        return x1, y1

    # Case 3: One side < sq_size
    else:
        if w < sq_size:  # 宽度较短
            x1 = random.randint(x - sq_size + w, x)  # 左上角的 x1 随机
            y1 = random.randint(y, y + h - sq_size)  # 高度随机分布
        else:  # 高度较短
            x1 = random.randint(x, x + w - sq_size)  # 宽度随机分布
            y1 = random.randint(y - sq_size + h, y)  # 左上角的 y1 随机
        return x1, y1

def remap_points(annitem):
    points = annitem['points']
    if len(points) < 2:
        return annitem
    p1_x,p1_y = points[0]['x'],points[0]['y']
    p2_x,p2_y = points[1]['x'],points[1]['y']
    if p1_x < p2_x and p1_y < p2_y:
        return annitem
    if p1_x > p2_x and p1_y > p2_y:
        annitem['points'] = [points[1], points[0]]
        annitem['region'] = dict(
            x = p2_x, y = p2_y,
            width = p1_x - p2_x, height = p1_y - p2_y
        )
        return annitem
    
    return None


def read_json_anno(json_path):
    import chardet

    with open(json_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open(json_path,'r',encoding = encoding) as f:
        data = json.load(f)
    annotations = data['annotation']

    return annotations

def read_json_valid(json_path, max_xy):
    import chardet
    
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
