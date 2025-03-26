from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import warnings
import os
import random
import json
import multiprocessing
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt
from cerwsi.utils import (KFBSlide,remap_points,read_json_anno,decode_xml,random_cut_square,
                          is_bbox_inside,overlap_enough,draw_OD)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

LEVEL = 1
PATCH_EDGE = 700
STRIDE = 650
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

def find_matching_bboxes(target_bbox, grid_size = PATCH_EDGE, stride = STRIDE, min_overlap=50):
    matching_bboxes = []
    x1_min, y1_min, x1_max, y1_max = target_bbox

    # 计算col 和 row 范围
    col_min = int(max(0, x1_min // stride))
    col_max = int(x1_max // stride)
    row_min = int(max(0, y1_min // stride))
    row_max = int(y1_max // stride)
    
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x2_min = col * stride
            y2_min = row * stride
            x2_max = x2_min + grid_size
            y2_max = y2_min + grid_size
            bbox = [x2_min, y2_min, x2_max, y2_max]
            # 判断完全包含 (target_bbox完全在该patch内部)
            if is_bbox_inside(target_bbox, bbox):
                relative_bbox = [x1_min - x2_min, y1_min - y2_min, x1_max - x2_min, y1_max - y2_min]
                matching_bboxes.append(([row,col], relative_bbox))
                continue

            # 计算交集区域
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                inter_w = inter_x_max - inter_x_min
                inter_h = inter_y_max - inter_y_min
                if inter_w > min_overlap and inter_h > min_overlap:
                    relative_bbox = [inter_x_min - x2_min, inter_y_min - y2_min, inter_x_max - x2_min, inter_y_max - y2_min]
                    matching_bboxes.append(([row,col], relative_bbox))

    return matching_bboxes

def filter_pos_bbox(annos):
    all_pos_bbox = []
    for ann_ in annos:
        ann = remap_points(ann_)
        if ann is None:
            continue
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if w >20 and h>20 and sub_class in POSITIVE_CLASS:
            all_pos_bbox.append([x, y, x+w, y+h])
    return all_pos_bbox

def ensure_neg(square, pos_bbox):
    for pbox in pos_bbox:
        if is_bbox_inside(pbox, square, tolerance=10):
            return False
        if overlap_enough(pbox, square, min_overlap=25):
            return False
    return True

def process_pos_slide(rowInfo):
    pos_patches_result = {} # key is patch id, value is patch anno info
    neg_patches_result = []
    slide = KFBSlide(f'{args.data_root_dir}/{rowInfo.kfb_path}')
    swidth, sheight = slide.level_dimensions[LEVEL]
    downsample_ratio = slide.level_downsamples[LEVEL]
    total_cols = int(swidth // STRIDE) + 1
    json_path = f'{args.data_root_dir}/{rowInfo.anno_path}'
    annos = read_json_anno(json_path)
    pos_bbox = filter_pos_bbox(annos)
    neg_patch_idx = 0
    for ann_ in annos:
        ann = remap_points(ann_)
        if ann is None:
            continue
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        
        origin_edge = PATCH_EDGE*downsample_ratio
        if w>origin_edge and h>origin_edge and sub_class == 'ROI':
            x1,y1 = random_cut_square((x,y,w,h), origin_edge)
            square = [x1,y1,x1+origin_edge,y1+origin_edge]
            if ensure_neg(square, pos_bbox):
                filename = f"{rowInfo.patientId}_{neg_patch_idx}.png"
                neg_patches_result.append({
                    'filename': filename,
                    'square_x1y1': [x1//downsample_ratio,y1//downsample_ratio],
                    'bboxes': [],
                    'clsnames': [],
                    'diagnose': 0
                })
                neg_patch_idx += 1
                continue
                
        if w <=20 or h<=20 or sub_class not in POSITIVE_CLASS:
            continue

        scaled_bbox = np.array([x,y,x+w,y+h]) // downsample_ratio
        target_bbox = scaled_bbox.tolist()
        match_patches = find_matching_bboxes(target_bbox, min_overlap=25)
        for stp,coord in match_patches:
            row,col= stp
            patchid = row * total_cols + col
            
            sx1,sy1 = col * STRIDE, row * STRIDE
            sx2,sy2 = sx1 + PATCH_EDGE, sy1 + PATCH_EDGE
            if sx2 > swidth:
                exceed = sx2 - swidth
                sx1 = sx1 - exceed
                sx2 = swidth
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1 - exceed,iy1,ix2 - exceed,iy2]
            if sy2 > sheight:
                exceed = sy2 - sheight
                sy1 = sy1 - exceed
                sy2 = sheight
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1,iy1 - exceed,ix2,iy2 - exceed]

            if patchid not in pos_patches_result.keys():
                pos_patches_result[patchid] = {
                    'filename':f'{rowInfo.patientId}_{patchid}.png',
                    'square_x1y1': (sx1,sy1),
                    'bboxes': [coord],
                    'clsnames': [sub_class],
                    'diagnose': 1
                }
            else:
                pos_patches_result[patchid]['bboxes'].append(coord)
                pos_patches_result[patchid]['clsnames'].append(sub_class)
    
    return list(pos_patches_result.values()),neg_patches_result

def process_pos_slide_wxl(rowInfo):
    patches_result = {} # key is patch id, value is patch anno info
    slide = KFBSlide(f'{args.data_root_dir}/{rowInfo.kfb_path}')
    swidth, sheight = slide.level_dimensions[LEVEL]
    downsample_ratio = slide.level_downsamples[LEVEL]
    total_cols = int(swidth // STRIDE) + 1
    
    anno_path = f'{args.data_root_dir}/{rowInfo.anno_path}'
    all_rects = decode_xml(anno_path)
    for x1,y1,x2,y2 in all_rects:
        w,h = x2-x1, y2-y1
        sub_class = rowInfo.kfb_clsname

        if w <=20 or h<=20:
            continue
        scaled_bbox = np.array([x1,y1,x2,y2]) // downsample_ratio
        target_bbox = scaled_bbox.tolist()
        match_patches = find_matching_bboxes(target_bbox, min_overlap=25)
        for stp,coord in match_patches:
            row,col= stp
            patchid = row * total_cols + col
            
            sx1,sy1 = col * STRIDE, row * STRIDE
            sx2,sy2 = sx1 + PATCH_EDGE, sy1 + PATCH_EDGE
            if sx2 > swidth:
                exceed = sx2 - swidth
                sx1 = sx1 - exceed
                sx2 = swidth
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1 - exceed,iy1,ix2 - exceed,iy2]
            if sy2 > sheight:
                exceed = sy2 - sheight
                sy1 = sy1 - exceed
                sy2 = sheight
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1,iy1 - exceed,ix2,iy2 - exceed]

            if patchid not in patches_result.keys():
                patches_result[patchid] = {
                    'filename':f'{rowInfo.patientId}_{patchid}.png',
                    'square_x1y1': (sx1,sy1),
                    'bboxes': [coord],
                    'clsnames': [sub_class],
                    'diagnose': 1
                }
            else:
                patches_result[patchid]['bboxes'].append(coord)
                patches_result[patchid]['clsnames'].append(sub_class)
    
    return list(patches_result.values()),[]

def gene_patch_json():
    train_data_df = pd.read_csv(args.train_csv_file)
    val_data_df = pd.read_csv(args.val_csv_file)

    for data_df,mode in zip([train_data_df,val_data_df], ['train','val']):
        all_patch_list = []
        total_pos_nums, total_neg_nums = 0,0
        for row in tqdm(data_df.itertuples(index=True), total=len(data_df)):
            # if row.Index > 5:
            #     break
            if row.kfb_clsname != 'NILM':
                if row.kfb_source == 'WXL_1':
                    pos_patch_list,neg_patch_list = process_pos_slide_wxl(row)
                else:
                    pos_patch_list,neg_patch_list = process_pos_slide(row)
                total_pos_nums += len(pos_patch_list)
                total_neg_nums += len(neg_patch_list)
                patch_list = [*pos_patch_list, *neg_patch_list]
                
                all_patch_list.append({
                    'patientId': row.patientId,
                    'kfb_path': row.kfb_path,
                    'patch_list': patch_list
                })
        print(f'{mode}: {total_pos_nums} pos patches and {total_neg_nums} neg patches.')

        with open(f'{args.save_dir}/annofiles/{mode}_posslide_patches.json', 'w') as f:
            json.dump(all_patch_list, f)

def cut_save(kfb_list):
    os.makedirs(f'{args.save_dir}/images/Pos', exist_ok=True)
    os.makedirs(f'{args.save_dir}/images/NegInPos', exist_ok=True)
    for kfbinfo in tqdm(kfb_list, ncols=80):
        slide = KFBSlide(f'{args.data_root_dir}/{kfbinfo["kfb_path"]}')
        patch_list = kfbinfo['patch_list']
        for patchinfo in patch_list:
            x1,y1 = patchinfo['square_x1y1']
            location, level, size = (x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            if patchinfo['diagnose'] == 0:
                path_prefix = f'{args.save_dir}/images/NegInPos'
            else:
                path_prefix = f'{args.save_dir}/images/Pos'
            read_result.save(f'{path_prefix}/{patchinfo["filename"]}')

def cut_patch():
    for mode in ['train', 'val']:
        with open(f'{args.save_dir}/annofiles/{mode}_posslide_patches.json', 'r') as f:
            kfb_list = json.load(f)
        
        cpu_num = 8
        set_split = np.array_split(kfb_list, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(cut_save,(set_group,))
            processes.append(p)

        for p in processes:
            p.get()

def vis_sample(vis_nums):
    sample_save_dir = 'statistic_results/0319/cut_pos_sample'
    os.makedirs(sample_save_dir, exist_ok=True)
    with open(f'{args.save_dir}/annofiles/train_posslide_patches.json', 'r') as f:
        kfb_list = json.load(f)
    
    for idx, kfbinfo in enumerate(kfb_list):
        if idx == vis_nums:
            break
        slide = KFBSlide(f'{args.data_root_dir}/{kfbinfo["kfb_path"]}')
        patch_list = kfbinfo['patch_list']
        random.shuffle(patch_list)
        for pidx, patchinfo in enumerate(patch_list):
            if pidx == vis_nums:
                break
            x1,y1 = patchinfo['square_x1y1']
            innerbbox,bbox_clsname = patchinfo['bboxes'],patchinfo['clsnames']

            inside_items = []
            for coords,clsname in zip(innerbbox,bbox_clsname):
                cx1,cy1,cx2,cy2 = coords
                inside_items.append({
                    'sub_class': clsname,
                    'region': dict(x=cx1,y=cy1,width=cx2-cx1,height=cy2-cy1)
                })
            
            location, level, size = (x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            filename = patchinfo["filename"]
            square_coords = [0,0,PATCH_EDGE,PATCH_EDGE]
            draw_OD(read_result, f'{sample_save_dir}/{filename}', square_coords, inside_items,category_colors)

            

parser = argparse.ArgumentParser()
parser.add_argument('train_csv_file', type=str)
parser.add_argument('val_csv_file', type=str)
parser.add_argument('--valid_model_ckpt', type=str)
parser.add_argument('--data_root_dir', type=str, default='/medical-data/data')
parser.add_argument('--save_dir', type=str) # {save_dir}/images {save_dir}/annofiles

args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(f'{args.save_dir}/images', exist_ok=True)
    os.makedirs(f'{args.save_dir}/annofiles', exist_ok=True)
    # gene_patch_json()
    # vis_sample(vis_nums = 5)
    cut_patch()

'''
python scripts/0319/slide_cut_pos.py \
    data_resource/slide_anno/0319/train.csv \
    data_resource/slide_anno/0319/val.csv \
    --save_dir /nfs5/zly/codes/CerWSI/data_resource/0319

train: 54028 pos patches and 11283 neg patches.
val: 14199 pos patches and 2886 neg patches.
'''