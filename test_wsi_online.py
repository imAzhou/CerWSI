import torch
import json
import time
from mmpretrain.structures import DataSample
import argparse
import cv2
from mmengine.config import Config
from math import ceil
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import warnings
import os
import copy
from torchvision import transforms
from mmengine.logging import MMLogger
from cerwsi.utils import (KFBSlide, set_seed,)
from cerwsi.nets import ValidClsNet, PatchClsNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

LEVEL = 1
PATCH_EDGE = 700
CERTAIN_THR = 0.7
NEGATIVE_THR = 0.5
positive_ratio_thr = 0.05

def inference_valid_batch(valid_model, read_result_pool, curent_id, save_prefix):
    data_batch = dict(inputs=[], data_samples=[])
    for read_result in read_result_pool:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
    
    valid_idx = []
    for idx,pred_output in enumerate(outputs):
        o_img = read_result_pool[idx]
        if max(pred_output.pred_score) > CERTAIN_THR:
            if pred_output.pred_label == 1:
                n_tag = 'valid'
                curent_id[1] += 1
                valid_idx.append(idx)
            else:
                n_tag = 'invalid'
                curent_id[0] += 1
        else:
            n_tag = 'uncertain'
            curent_id[2] += 1
        
        if args.visual_pred and n_tag in args.visual_pred:
            new_save_prefix = save_prefix.replace('valid_tag', n_tag)
            folder_path = os.path.dirname(new_save_prefix)
            os.makedirs(folder_path, exist_ok=True)
            o_img.save(f'{new_save_prefix}_{sum(curent_id)}.png')

    return valid_idx

def inference_batch_pn(pn_model, valid_input, save_prefix):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_inputs = [transform(read_result) for read_result in valid_input]
    images_tensor = torch.stack(img_inputs, dim=0)
    data_batch = dict(images=images_tensor)

    with torch.no_grad():
        outputs = pn_model(data_batch, 'val')
    if outputs['img_probs'].numel() == 1:
        outputs['img_probs'] = outputs['img_probs'].unsqueeze(0)
    pred_result = []
    for idx,pred_output in enumerate(outputs['img_probs']):
        o_img = valid_input[idx]
        pred_clsid = 1 if int(pred_output > NEGATIVE_THR) else 0
        # 0/1，img_probs，*cls_pos_probs
        res_arr = [pred_clsid, round(pred_output.detach().item(), 6)]
        if 'pos_probs' in outputs:
            res_arr.extend([round(conf.detach().item(), 6) for conf in outputs['pos_probs'][idx]])
        pred_result.append(res_arr)
        if args.visual_pred and str(pred_clsid) in args.visual_pred:
            timestamp = time.time()
            os.makedirs(f'{save_prefix}/{pred_clsid}', exist_ok=True)
            o_img.save(f'{save_prefix}/{pred_clsid}/{timestamp}.png')
    
    attn_array = None
    if 'attn_array' in outputs:
        attn_array = outputs['attn_array'].detach().cpu()   # (bs, layer, num_cls, num_tokens)
    return pred_result,attn_array

def process_patches(proc_id, start_points, valid_model, pn_model, kfb_path, patientId):   
    save_root_dir = f'predict_results/{patientId}'
    save_prefix = f'{save_root_dir}/valid_tag/{patientId}_c{proc_id}'

    if args.visual_pred is not None:
        for tag in args.visual_pred:
            save_dir = f'{save_root_dir}/{tag}'
            os.makedirs(save_dir, exist_ok=True)
    
    slide = KFBSlide(kfb_path)
    downsample_ratio = slide.level_downsamples[LEVEL]
    read_result_pool, valid_read_result = [], []
    curent_id = [0,0,0]
    total_pred_results = []
    
    for p_idx,(x,y)in enumerate(start_points):
        location, level, size = (x, y), LEVEL, (PATCH_EDGE, PATCH_EDGE)
        read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
        coords = np.array([x, y, x+PATCH_EDGE, y+PATCH_EDGE])*downsample_ratio
        read_result_pool.append({
            'image': read_result,
            'coords': coords.tolist(),   # patch 坐标 (在 LEVEL=0上的坐标)
            'attn_array': [], # 该patch块的阴阳热力值（仅valid patch有此值）
            'pn_pred': [],  # 该patch块的阳性预测概率（仅valid patch有此值）
        })
        
        if len(read_result_pool) % args.test_bs == 0 or p_idx == len(start_points)-1:
            imgs = [item['image'] for item in read_result_pool]
            valid_idx = inference_valid_batch(valid_model, imgs, curent_id, save_prefix)
            valid_read_result.extend([read_result_pool[idx] for idx in valid_idx])
            read_result_pool = []
            print(f'\rCore: {proc_id}, 当前已处理: {sum(curent_id)}', end='')
        
        if len(valid_read_result) > 0 and not args.only_valid:
            imgs = [item['image'] for item in valid_read_result]
            pred_result,attn_array = inference_batch_pn(pn_model, imgs, save_prefix)
            for pidx, pitem in enumerate(valid_read_result):
                # 只保存第二次注意力计算后的注意力值
                if attn_array is not None:
                    # attn_array.shape: (bs, layer, num_classes, num_tokens)
                    pitem['attn_array'] = attn_array[pidx][1].tolist()
                pitem['pn_pred'] = pred_result[pidx]
                del pitem['image']
            total_pred_results.extend(valid_read_result)
            valid_read_result = []

    print(f'Core: {proc_id}, process {sum(curent_id)} patches done!!')
    return curent_id, total_pred_results

def get_pn_model(device):
    cfg = Config.fromfile(args.config_file)
    cfg.backbone_ckpt = None
    model = PatchClsNet(cfg).to(device)
    model.load_ckpt(args.ckpt)
    model.eval()
    return model

def multiprocess_inference():
    all_kfb_info = pd.read_csv(args.test_csv_file)

    device = torch.device(args.device)
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(args.valid_model_ckpt))

    if args.only_valid:
        pn_model = None
    else:
        pn_model = get_pn_model(device)

    print('='*10 + 'Models Load Done!' + '='*10)
    if args.record_save_dir:
        os.makedirs(args.record_save_dir, exist_ok=True)
    logger = MMLogger.get_instance('test_wsi', log_file=f'{args.record_save_dir}/test_wsi.log')

    low_valid_kfb_info = []
    for row in all_kfb_info.itertuples(index=True):
        start_time = time.time()
        print('collecting start points... ')
        slide = KFBSlide(row.kfb_path)
        width, height = slide.level_dimensions[LEVEL]
        iw, ih = ceil(width/PATCH_EDGE), ceil(height/PATCH_EDGE)
        r2 = (int(max(iw, ih)*1.1)//2)**2
        cix, ciy = iw // 2, ih // 2
        slide_start_points = []
        for j, y in enumerate(range(0, height, PATCH_EDGE)):
            for i, x in enumerate(range(0, width, PATCH_EDGE)):
                if (i-cix)**2 + (j-ciy)**2 > r2:
                    continue
                slide_start_points.append((x, y))
        print(f'total start points: {len(slide_start_points)}')
        
        cpu_num = args.cpu_num
        set_split = np.array_split(slide_start_points, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        # pool_state = Manager().dict()
        # pool_state["valid_cnt"] = [0,0,0]
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(process_patches,
                                    (proc_id, set_group, valid_model, pn_model,
                                     row.kfb_path, row.patientId))
            processes.append(p)

        valid_result, pn_result = [], []
        for p in processes:
            valids,pns = p.get()
            valid_result.append(valids)
            pn_result.extend(pns)
        workers.close()
        workers.join()

        t_delta = time.time() - start_time
        curent_id = np.array(valid_result)

        if args.only_valid:
            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}, total: {len(slide_start_points)}')
        else:
            confi_pred = np.array([res['pn_pred'] for res in pn_result])
            if len(confi_pred) == 0: # 无任何 valid image
                pn_result_patch_clsid = []
                pred_confi = []
            else:
                pn_result_patch_clsid = confi_pred[:,0]
                pred_confi = [f'{" ".join(map(str, conf.tolist()))} \n' for conf in confi_pred[:, 1:]]

            # 打印每张 slide 阴阳 patch 预测结果
            p_path_num = int(np.sum(pn_result_patch_clsid))
            n_patch_num = len(pn_result_patch_clsid) - p_path_num
            p_ratio = p_path_num / (p_path_num + n_patch_num + 1e-6)    # 防止除0
            pred_clsid = int(p_ratio > positive_ratio_thr)
            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}(positive:{p_path_num} negative:{n_patch_num} p_ratio:{p_ratio:0.4f} pred/gt:{pred_clsid}/{row.kfb_clsid}-{row.kfb_clsname})')

            # 保存每张patch在每个类别上的预测置信度
            posi_confi_save_dir = f'{args.record_save_dir}/posi_conf'
            os.makedirs(posi_confi_save_dir, exist_ok=True)
            with open(f'{posi_confi_save_dir}/{row.patientId}.txt', 'w') as f:
                f.writelines(pred_confi)

            # 保存每张patch在每个类别上的响应值（热力图）
            heat_save_dir = f'{args.record_save_dir}/heat_value'
            os.makedirs(heat_save_dir, exist_ok=True)
            with open(f'{heat_save_dir}/{row.patientId}.json', 'w') as f:
                json.dump(pn_result, f)

        if np.sum(curent_id[:,1]) <= 1000 or np.sum(curent_id[:,0]) > np.sum(curent_id[:,1])*2:
            low_valid_kfb_info.append([row.kfb_path, row.kfb_clsid, row.kfb_clsname, row.patientId, row.kfb_source])

    df_low_valid = pd.DataFrame(low_valid_kfb_info, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_low_valid.to_csv(f'{args.record_save_dir}/low_valid.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('test_csv_file', type=str)
parser.add_argument('valid_model_ckpt', type=str)
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--only_valid', action='store_true')
parser.add_argument('--visual_pred', type=str, nargs='*', choices=['0', '1', 'invalid', 'valid', 'uncertain'])
parser.add_argument('--cpu_num', type=int, default=1, help='multiprocess cpu num')
parser.add_argument('--test_bs', type=int, default=16, help='batch size of model test')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    multiprocessing.set_start_method('spawn', force=True)
    
    multiprocess_inference()

'''
Time of process kfb elapsed: 805.35 seconds, valid: 6126, invalid: 1108, uncertain: 72, total: 7306
Time of process kfb elapsed: 71.05 seconds, valid: 6126, invalid: 1108,  uncertain: 72, total: 7306

CUDA_VISIBLE_DEVICES=0 python test_wsi_online.py \
    data_resource/debug2.csv \
    checkpoints/vlaid_cls_best.pth \
    log/l_cerscan_v2/wscernet/2025_03_25_11_08_39/config.py \
    log/l_cerscan_v2/wscernet/2025_03_25_11_08_39/checkpoints/best.pth \
    --record_save_dir log/debug_ours \
    --cpu_num 8 \
    --test_bs 128 \
    --visual_pred 1
    --only_valid \
    --visual_pred invalid valid 1
'''