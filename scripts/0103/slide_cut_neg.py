import os
import time
import pandas as pd
import torch
from cerwsi.nets import ValidClsNet
from cerwsi.utils import KFBSlide
from PIL import Image
import random
import cv2
import numpy as np
from mmpretrain.structures import DataSample
import json
import warnings
from tqdm import tqdm

CERTAIN_THR = 0.7
PATCH_EDGE = 700
cut_nums_each = 150
kfb_root_dir = '/medical-data/data'
img_save_dir = 'data_resource/0103/images/Neg'
os.makedirs(img_save_dir, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

def process_patches(proc_id, set_group, valid_model, kfb_path, patientId):
    slide = KFBSlide(f'{kfb_root_dir}/{kfb_path}')

    patch_list = []
    for x1,y1 in set_group:
        read_result = Image.fromarray(slide.read_region((x1,y1), 0, (PATCH_EDGE,PATCH_EDGE)))
        data_batch = dict(inputs=[], data_samples=[])
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())
        data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
        with torch.no_grad():
            outputs = valid_model.val_step(data_batch)
    
        if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
            filename = f'{patientId}_c{proc_id}p{len(patch_list)}.png'
            patch_list.append({
                'filename': filename,
                'square_x1y1': (int(x1),int(y1)),
                'bboxes': [],
                'clsnames': [],
                'diagnose': 0,
                'gtmap_14': (np.zeros((14,14), dtype=int)).tolist()
            })
            read_result.save(f'{img_save_dir}/{filename}')
    return patch_list

def cut_random_neg():
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/vlaid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))

    anno_save_dir = 'data_resource/0103/annofiles'

    train_csv = pd.read_csv('data_resource/ROI/annofile/1223_train.csv')
    val_csv = pd.read_csv('data_resource/ROI/annofile/1223_val.csv')
    filtered = {
        'train': train_csv[train_csv['kfb_clsid'] == 0],
        'val': val_csv[val_csv['kfb_clsid'] == 0],
    }

    for mode in ['train','val']:
        all_patch_list = []
        total_patches = 0

        txt_records = []
        low_valid_records = []
        total_nums = len(filtered[mode])
        filtered[mode] = filtered[mode].reset_index(drop=True)
        for r_idx, row in filtered[mode].iterrows():
            kfb_path, patientId = row['kfb_path'], row['patientId']
            slide = KFBSlide(f'{kfb_root_dir}/{kfb_path}')
            max_x, max_y = slide.level_dimensions[0]
            slide_patch_list = []
            start_time = time.time()
            for i in tqdm(range(cut_nums_each)):
                x1,y1 = random.randint(0, max_x-PATCH_EDGE),random.randint(0, max_y-PATCH_EDGE)
                read_result = Image.fromarray(slide.read_region((x1,y1), 0, (PATCH_EDGE,PATCH_EDGE)))
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
            
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    filename = f'{patientId}_{len(slide_patch_list)}.png'
                    slide_patch_list.append({
                        'filename': filename,
                        'square_x1y1': (int(x1),int(y1)),
                        'bboxes': [],
                        'clsnames': [],
                        'diagnose': 0,
                        'gtmap_14': (np.zeros((14,14), dtype=int)).tolist()
                    })
                    read_result.save(f'{img_save_dir}/{filename}')

            t_delta = time.time() - start_time
            print(f'[{r_idx+1}/{total_nums}] {patientId} cut nums: {len(slide_patch_list)}, cost: {t_delta:0.2f}s')
            txt_records.append(f'{patientId} cut nums: {len(slide_patch_list)}. \n')
            if len(slide_patch_list) < 30:
                low_valid_records.append(f'{patientId} cut nums: {len(slide_patch_list)}. \n')
            
            all_patch_list.append({
                'patientId': patientId,
                'kfb_path': kfb_path,
                'patch_list': slide_patch_list
            })
            total_patches += len(slide_patch_list)
        
        txt_records.append(f'{mode}: {total_patches} patches. \n')
        with open(f'{anno_save_dir}/{mode}_neg_patch_nums.txt', 'w') as f:
            f.writelines(txt_records)
            f.writelines(f"{'='*20} \n\n\n")
            f.writelines(low_valid_records)

        with open(f'{anno_save_dir}/{mode}_neg_patches.json', 'w') as f:
            json.dump(all_patch_list, f)

if __name__ == '__main__':
    cut_random_neg()