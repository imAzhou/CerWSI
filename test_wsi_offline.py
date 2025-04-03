import torch
import os
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from mmengine.dist import collect_results
import argparse
from mmengine.config import Config
from cerwsi.datasets import load_slide_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
import json
from collections import defaultdict
from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics,print_confusion_matrix,draw_OD

POSITIVE_THR = 0.5

parser = argparse.ArgumentParser()
# base args
parser.add_argument('test_csv_file', type=str)
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def test_net(cfg, model):
    valloader = load_slide_data(cfg)
    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    predict_rsults = []
    for idx, data_batch in enumerate(pbar):
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        
        for bidx in range(len(outputs['images'])):
            patientId = outputs['image_patientIds'][bidx]
            filename = os.path.basename(outputs['image_paths'][bidx])
            if 'img_probs' in outputs:
                pred_prob = outputs['img_probs'][bidx]
                pred_label = (pred_prob > POSITIVE_THR).int().item()
            elif 'token_classes' in outputs:
                pred_cls = torch.max(outputs['token_classes'][bidx], dim=-1)[0]
                pred_label = (pred_cls > 0).int().item()
            predict_rsults.append((patientId,filename,pred_label))

    results = collect_results(predict_rsults, len(valloader.dataset))
    if is_main_process():
        pbar.close()
        patient_dict = defaultdict(list)
        for patientId, filename, pred_label in results:
            patient_dict[patientId].append([filename, pred_label])
        patient_dict = dict(patient_dict)
        
        with open(f'{args.save_dir}/slide_pred_result.json', 'w') as f:
            json.dump(patient_dict, f)
        
        evaluate_slide(patient_dict)

def evaluate_slide(patient_dict):
    all_kfb_info = pd.read_csv(args.test_csv_file)
    positive_ratio_thr = 0.05
    y_true,y_pred = [],[]
    for row in tqdm(all_kfb_info.itertuples(), total=len(all_kfb_info), ncols=80):
        pos_pred = [i[1] for i in patient_dict[row.patientId]]
        p_path_num = sum(pos_pred)
        n_patch_num = len(pos_pred) - p_path_num
        p_ratio = p_path_num / (p_path_num + n_patch_num + 1e-6)    # 防止除0
        pred_clsid = int(p_ratio > positive_ratio_thr)
        
        y_true.append(row.kfb_clsid)
        y_pred.append(pred_clsid)
        
    metric_result = calculate_metrics(y_true,y_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)
    print_confusion_matrix(cm)


def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    
    model = PatchClsNet(cfg).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_without_ddp.load_ckpt(args.ckpt)
    test_net(cfg, model)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
    # analyze(f'{args.save_dir}/pred_results_0.5.json')
    # with open(f'{args.save_dir}/slide_pred_result.json', 'r') as f:
    #     patient_dict = json.load(f)
    # evaluate_slide(patient_dict)

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12341 test_wsi_offline.py \
    /c22073/zly/datasets/CervicalDatasets/LCerScanv2/annofiles/val.csv \
    log/l_cerscan_v2/wscer_partial/2025_04_01_17_04_05/config.py \
    log/l_cerscan_v2/wscer_partial/2025_04_01_17_04_05/checkpoints/best.pth \
    log/l_cerscan_v2/wscer_partial/2025_04_01_17_04_05
'''