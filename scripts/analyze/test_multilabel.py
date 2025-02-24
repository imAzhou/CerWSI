import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from cerwsi.datasets import TokenClsDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from mmengine.dist import collect_results
import argparse
from mmengine.config import Config
# from cerwsi.nets import MultiPatchUNI
from cerwsi.nets import CerMCNet
from cerwsi.utils import MyMultiTokenMetric,MultiPosMetric
from cerwsi.utils import set_seed, init_distributed_mode, build_evaluator,is_main_process
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics,print_confusion_matrix,draw_OD

POSITIVE_THR = 0.5
POSITIVE_CLASS = ['ASC-US','LSIL', 'ASC-H', 'HSIL', 'AGC']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def draw_pred(img_item):
    img_path = img_item['img_path']
    img = Image.open(img_path)
    h,w = img.size
    filename = os.path.basename(img_path)
    square_coords = [0,0,w,h]
    inside_items = []
    scale_ratio = h//14
    for tk in img_item['token_labels']:
        row,col,clsid = tk
        y1,x1 = row*scale_ratio,col*scale_ratio
        clsname = POSITIVE_CLASS[clsid-1]
        inside_items.append(
            dict(sub_class=clsname, region=dict(x=x1,y=y1,width=scale_ratio,height=scale_ratio))
        )
    draw_OD(img, f'{args.save_dir}/FN/{filename}', square_coords, inside_items, category_colors)

def load_data(cfg):
    def custom_collate(batch):
        # 拆分 batch 中的图像和标签
        images = [item[0] for item in batch]  # 所有 image_tensor，假设 shape 一致
        image_labels = [item[1] for item in batch]
        image_paths = [item[3] for item in batch]
        token_labels = [item[2] for item in batch]

        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)

        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'token_labels': token_labels,  # 保持 label 的原始列表形式
            'image_paths': image_paths
        }

    train_dataset = TokenClsDataset(cfg.data_root, 'train')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                            pin_memory=True,
                            batch_size=cfg.train_bs, 
                            sampler = train_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    val_dataset = TokenClsDataset(cfg.data_root, 'val')
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            pin_memory=True,
                            batch_size=cfg.val_bs, 
                            sampler = val_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    
    return train_loader, val_loader

def test_net(cfg, model):
    trainloader,valloader = load_data(cfg)
    evaluator = build_evaluator([MyMultiTokenMetric(thr=POSITIVE_THR)])
    # evaluator = build_evaluator([MultiPosMetric(thr=POSITIVE_THR)])

    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    predict_rsults = []
    for idx, data_batch in enumerate(pbar):
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        
        evaluator.process(data_samples=[outputs], data_batch=None)
        for bidx in range(len(outputs['img_probs'])):
            pred_label = (outputs['img_probs'][bidx] > POSITIVE_THR).int().item()
            pos_pred = (outputs['pos_probs'][bidx] > POSITIVE_THR).int().cpu().tolist()
            if pred_label == 0:
                pred_multi_label = [0]
            else:
                pred_multi_label = [clsidx+1 for clsidx,pred in enumerate(pos_pred) if pred == 1]
            
            pos_gt = [0]
            if len(outputs['token_labels'][bidx]) > 0:
                pos_gt = list(set([tk[-1] for tk in outputs['token_labels'][bidx]]))
            result = dict(
                img_path = outputs['image_paths'][bidx],
                gt_label = outputs['image_labels'][bidx].item(),
                pred_label = pred_label,
                pred_score = outputs['img_probs'][bidx].item(),
                token_labels = outputs['token_labels'][bidx],
                pos_gt = pos_gt,
                pos_pred = pred_multi_label
            )
            predict_rsults.append(result)
    
    metrics = evaluator.evaluate(len(valloader.dataset))   
    results = collect_results(predict_rsults, len(valloader.dataset))
    if is_main_process():
        pbar.close()
        print(metrics)
        pred_results_dict = {'results':results}
        json_path = f'{args.save_dir}/pred_results_{POSITIVE_THR}.json'
        with open(json_path, 'w') as f:
            json.dump(pred_results_dict, f)
        # analyze(json_path)

def analyze(json_path):
    with open(json_path, 'r') as f:
        pred_results = json.load(f)
    
    y_true,y_pred = [],[]
    conflict_pred = 0
    error_pos_cls = [0]*len(POSITIVE_CLASS)
    for imgitem in tqdm(pred_results['results']):
        y_true.append(imgitem['gt_label'])
        y_pred.append(imgitem['pred_label'])
        if imgitem['pred_label'] == 1 and len(imgitem['pos_pred']) == 0:
            conflict_pred += 1
        if imgitem['gt_label'] == 1 and imgitem['pred_label'] == 0:
            # os.makedirs(f'{args.save_dir}/FN',exist_ok=True)
            # draw_pred(imgitem)
            tks = [tk[-1]-1 for tk in imgitem['token_labels']]
            for i in range(len(error_pos_cls)):
                if i in tks:
                    error_pos_cls[i] += 1
    error_cls_table = PrettyTable()
    error_cls_table.field_names = POSITIVE_CLASS
    error_cls_table.add_row(error_pos_cls)
    print(error_cls_table)

    metric_result = calculate_metrics(y_true,y_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)

    print_confusion_matrix(cm)
    print(f'conflict_pred: {conflict_pred}')

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    
    model = CerMCNet(
        num_classes = cfg['num_classes'], 
        backbone_type = cfg.backbone_type,
        use_lora=cfg.use_lora
    ).to(device)
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

'''
CUDA_VISIBLE_DEVICES=0,1,2 torchrun  --nproc_per_node=3 --master_port=12340 scripts/analyze/test_multilabel.py \
    log/cdetector_ours/2025_02_20_23_12_21/config.py \
    log/cdetector_ours/2025_02_20_23_12_21/checkpoints/best.pth \
    log/cdetector_ours/2025_02_20_23_12_21

    
+--------+------+-------+------+-----+
| ASC-US | LSIL | ASC-H | HSIL | AGC |
+--------+------+-------+------+-----+
|  343   | 182  |   64  |  8   | 180 |
+--------+------+-------+------+-----+
+----------+-------------+-------------+
| accuracy | sensitivity | specificity |
+----------+-------------+-------------+
|  0.9355  |    0.953    |    0.9214   |
+----------+-------------+-------------+
+-----------------------------+
|       confusion matrix      |
+-----+-------+-------+-------+
|     |   0   |   1   |  sum  |
+-----+-------+-------+-------+
|  0  | 18685 |  1593 | 20278 |
|  1  |  761  | 15441 | 16202 |
| sum | 19446 | 17034 | 36480 |
+-----+-------+-------+-------+
conflict_pred: 578
'''