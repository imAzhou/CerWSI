import torch
from tqdm import tqdm
import time
from mmpretrain.structures import DataSample
import argparse
import cv2
from math import ceil
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import warnings
import os
import copy
from mmengine.logging import MMLogger
from cerwsi.utils import (KFBSlide, set_seed,)
from cerwsi.nets import ValidClsNet, PatchClsNet, PatchClsDINO

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

PATCH_EDGE = 500
CERTAIN_THR = 0.7
NEGATIVE_THR = 0.7
positive_ratio_thr = 0.005

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
    
    valid_input = []
    for idx,pred_output in enumerate(outputs):
        o_img = read_result_pool[idx]
        if max(pred_output.pred_score) > CERTAIN_THR:
            if pred_output.pred_label == 1:
                n_tag = 'valid'
                curent_id[1] += 1
                valid_input.append(o_img)
            else:
                n_tag = 'invalid'
                curent_id[0] += 1
        else:
            n_tag = 'uncertain'
            curent_id[2] += 1
        
        if args.visual_pred and n_tag in args.visual_pred:
            new_save_prefix = save_prefix.replace('valid_tag', n_tag)
            o_img.save(f'{new_save_prefix}_{sum(curent_id)}.png')

    return valid_input

def inference_batch_pn(pn_model, valid_input, save_prefix):
    data_batch = dict(inputs=[], data_samples=[])
    for read_result in valid_input:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())
    
    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = pn_model.val_step(data_batch)
    pred_result = []
    for idx,pred_output in enumerate(outputs):
        o_img = valid_input[idx]
        pred_clsid = 0 if int(pred_output.pred_score[0] > NEGATIVE_THR) else 1
        pred_result.append([pred_clsid, *[round(conf.item(), 6) for conf in pred_output.pred_score]])
        if args.visual_pred and str(pred_clsid) in args.visual_pred:
            timestamp = time.time()
            o_img.save(f'{save_prefix}/{pred_clsid}/{timestamp}.png')
    return pred_result       


def process_patches(proc_id, start_points, valid_model, pn_model, kfb_path, patientId):
    
    save_root_dir = f'predict_results/w{PATCH_EDGE}/{patientId}'
    save_prefix = f'{save_root_dir}/valid_tag/{patientId}_c{proc_id}'

    if args.visual_pred is not None:
        for tag in args.visual_pred:
            save_dir = f'{save_root_dir}/{tag}'
            os.makedirs(save_dir, exist_ok=True)
    
    slide = KFBSlide(kfb_path)
    read_result_pool, valid_read_result = [], []
    curent_id = [0,0,0]
    pn_pred_results = []
    for (x,y) in start_points:
        location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
        read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
        read_result_pool.append(read_result)
        
        if len(read_result_pool) % args.test_bs == 0:
            valid_input = inference_valid_batch(
                valid_model, read_result_pool, curent_id, save_prefix)
            read_result_pool = []
            valid_read_result.extend(valid_input)
            print(f'\rCore: {proc_id}, 当前已处理: {sum(curent_id)}', end='')
        
        if len(valid_read_result) > 0 and not args.only_valid:
            pred_result = inference_batch_pn(pn_model, valid_read_result, save_root_dir)
            pn_pred_results.extend(pred_result)
            valid_read_result = []
    
    if len(read_result_pool) > 0:
        valid_input = inference_valid_batch(
            valid_model, read_result_pool, curent_id, save_prefix)
        if len(valid_input) > 0 and not args.only_valid:
            pred_result = inference_batch_pn(pn_model, valid_input, save_root_dir)
            pn_pred_results.extend(pred_result)

    del read_result_pool
    torch.cuda.empty_cache()
    print(f'Core: {proc_id}, process {sum(curent_id)} patches done!!')
    return curent_id, pn_pred_results

def get_pn_model(device):
    if args.pn_model_type == 'resnet50':
        pn_model = PatchClsNet(num_classes = args.num_classes)
    elif args.pn_model_type == 'dinov2_s':
        pn_model = PatchClsDINO(num_classes = args.num_classes, device=device)
    
    pn_model.to(device)
    pn_model.eval()
    pn_state_dict = torch.load(args.pn_model_ckpt)
    if 'state_dict' in pn_state_dict:
        pn_state_dict = pn_state_dict['state_dict']
    pn_model.load_state_dict(pn_state_dict)

    return pn_model

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

    pred_kfb_info = []
    low_valid_kfb_info = []
    for row in all_kfb_info.itertuples(index=True):
        start_time = time.time()
        print('collecting start points... ')
        slide = KFBSlide(row.kfb_path)
        width, height = slide.level_dimensions[0]
        iw, ih = ceil(width/PATCH_EDGE), ceil(height/PATCH_EDGE)
        r2 = (int(max(iw, ih)*1.1)//2)**2
        cix, ciy = iw // 2, ih // 2
        slide_start_points = []
        for j, y in enumerate(range(0, height, PATCH_EDGE)):
            for i, x in enumerate(range(0, width, PATCH_EDGE)):
                if (i-cix)**2 + (j-ciy)**2 > r2:
                    continue
                slide_start_points.append((x, y))
        slide.close()
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
            pn_result = np.array(pn_result)
            pn_result_patch_clsid = pn_result[:,0]
            p_path_num = int(np.sum(pn_result_patch_clsid))
            n_patch_num = len(pn_result_patch_clsid) - p_path_num
            p_ratio = p_path_num / (p_path_num + n_patch_num)
            pred_clsid = int(p_ratio > positive_ratio_thr)
            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}(positive:{p_path_num} negative:{n_patch_num} p_ratio:{p_ratio:0.4f} pred/gt:{pred_clsid}/{row.kfb_clsid}-{row.kfb_clsname})')
            pred_kfb_info.append([row.kfb_path, row.kfb_clsid, p_path_num, n_patch_num])
            pred_confi = [f'{" ".join(map(str, conf.tolist()))} \n' for conf in pn_result[:, 1:]]
            posi_confi_save_dir = f'{args.record_save_dir}/posi_conf'
            os.makedirs(posi_confi_save_dir, exist_ok=True)
            with open(f'{posi_confi_save_dir}/{row.patientId}.txt', 'w') as f:
                f.writelines(pred_confi)

        if np.sum(curent_id[:,1]) <= 1000 or np.sum(curent_id[:,0]) > np.sum(curent_id[:,1])*2:
            low_valid_kfb_info.append([row.kfb_path, row.kfb_clsid, row.kfb_clsname, row.patientId, row.kfb_source])
    
    if not args.only_valid:
        df_pred = pd.DataFrame(pred_kfb_info, columns=['kfb_path', 'kfb_clsid', 'p_path_num', 'n_patch_num'])
        df_pred.to_csv(f'{args.record_save_dir}/pred_pn.csv', index=False)

    df_low_valid = pd.DataFrame(low_valid_kfb_info, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_low_valid.to_csv(f'{args.record_save_dir}/low_valid.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('test_csv_file', type=str)
parser.add_argument('valid_model_ckpt', type=str)
parser.add_argument('pn_model_type', type=str, choices=['resnet50', 'dinov2_s'])
parser.add_argument('pn_model_ckpt', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--num_classes', type=int, default=2)
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

python test_wsi.py \
    data_resource/train_neg.csv \
    checkpoints/vlaid_cls_best.pth \
    resnet50 \
    checkpoints/pn_cls_best/rcp_c6_v2.pth \
    --record_save_dir log/1127_train_neg \
    --num_classes 6 \
    --cpu_num 8 \
    --test_bs 64 \
    --visual_pred 1
    --only_valid \
    --visual_pred invalid valid 1
'''