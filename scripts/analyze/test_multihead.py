import torch
import os
from tqdm import tqdm
import torch.distributed as dist
import argparse
from mmengine.config import Config
from cerwsi.nets import PatchMultiHeadNet
from mmengine.dist import collect_results
from prettytable import PrettyTable
from cerwsi.utils import MyMultiLabelMetric,calculate_metrics,print_confusion_matrix
from cerwsi.utils import set_seed, init_distributed_mode, load_multilabel_dataset, build_evaluator,is_main_process
import json


POSITIVE_THR = 0.5
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

# parser = argparse.ArgumentParser()
# # base args
# parser.add_argument('dataset_config_file', type=str)
# parser.add_argument('--record_save_dir', type=str)
# parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--print_interval', type=int, default=10, help='random seed')
# parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
# parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

# args = parser.parse_args()

def test_net(cfg, model):
    trainloader,valloader = load_multilabel_dataset(
        d_config = cfg, seed = args.seed, distributed = args.distributed)
    
    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    predict_rsults = []
    for idx, data_batch in enumerate(pbar):
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        
        for data_sample in outputs:
            pred_true = data_sample.pred_score > POSITIVE_THR
            if sum(pred_true[1:]) == 0:
                # data_sample['pred_score'][0] > 0.7
                pred_label = torch.as_tensor([0]).to(pred_true.device)
            else:
                pred_label = torch.where(pred_true[1:])[0] + 1
            result = dict(
                img_path = data_sample.img_path,
                gt_label = data_sample.gt_label.cpu().tolist(),
                pred_label = pred_label.cpu().tolist(),
                prd_score = data_sample.pred_score.cpu().tolist()
            )
            predict_rsults.append(result)

    results = collect_results(predict_rsults, len(valloader.dataset))
    # metrics = evaluator.evaluate(len(valloader.dataset))
    if is_main_process():
        pbar.close()
        pred_results_dict = {'results':results}
        with open('analyze/multihead_pred_results.json', 'w') as f:
            json.dump(pred_results_dict, f)


def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    d_cfg = Config.fromfile(args.dataset_config_file)

    model = PatchMultiHeadNet(num_classes = d_cfg['num_classes'], device=device).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_weights = torch.load('log/multihead/2024_12_27_19_23_24/checkpoints/best.pth')
    model_without_ddp.load_state_dict(model_weights)
    test_net(d_cfg, model)

    if args.distributed:
        dist.destroy_process_group()

def analyze():
    with open('analyze/multihead_pred_results.json', 'r') as f:
        pred_results = json.load(f)
    
    y_true,y_pred = [],[]
    for imgitem in tqdm(pred_results['results']):
        pred_label = [idx for idx,value in enumerate(imgitem['prd_score']) if value > 0.5]
        gt_id = 1 if 0 not in imgitem['gt_label'] else 0
        pred_id = 1 if 0 not in pred_label else 0
        y_true.append(gt_id)
        y_pred.append(pred_id)
    metric_result = calculate_metrics(y_true,y_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)

    print_confusion_matrix(cm)
    

if __name__ == '__main__':
    # main()
    analyze()

'''
CUDA_VISIBLE_DEVICES=0,1,2 torchrun  --nproc_per_node=3 --master_port=12345 analyze/test_multihead.py \
    configs/dataset/multi_label_dataset.py
'''