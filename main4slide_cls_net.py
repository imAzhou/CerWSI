import torch
from tqdm import tqdm
from collections import OrderedDict
import argparse
from mmengine.config import Config
from cerwsi.nets import SlideClsNet
from cerwsi.utils import (set_seed, load_token_dataset, get_logger,get_train_strategy)
from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics
from sklearn.metrics import classification_report
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_net(cfg):
    model.train()
    trainloader = load_token_dataset('train', cfg)
    valloader = load_token_dataset('val', cfg)
    
    optimizer,lr_scheduler = get_train_strategy(model, cfg)

    logger, files_save_dir = get_logger(
        args.record_save_dir, model, cfg, 'slide_cls')
    
    for epoch in range(cfg.max_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(total=len(trainloader)*cfg.train_bs, desc=f'Train Epoch {epoch + 1}/{cfg.max_epochs}, LR: {current_lr:.6f}')
        for idx, data_batch in enumerate(trainloader):
            inputs = data_batch['inputs'].to(device)
            gt_label = data_batch['label'].to(device)
            loss = model.train_step(inputs, gt_label, optim_wrapper=optimizer)
            postfix = OrderedDict()
            postfix['loss'] = loss.item()
            pbar.set_postfix(postfix)
            pbar.update(cfg.train_bs)
            if idx % args.print_interval == 0:
                logger.info(f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}, loss: {loss.item():.6f}')
        
        pbar.close()
        lr_scheduler.step()

        model.eval()
        logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        pbar = tqdm(total=len(valloader)*cfg.val_bs, desc=f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        slide_pred, slide_gt = [],[]
        for idx, data_batch in enumerate(valloader):
            with torch.no_grad():
                inputs = data_batch['inputs'].to(device)
                pred_scores,pred_labels = model.val_step(inputs)
            
            slide_gt.extend(data_batch['label'].tolist())
            slide_pred.extend(pred_labels.squeeze(1).tolist())
            pbar.update(cfg.val_bs)
        pbar.close()
        metric_result = calculate_metrics(slide_gt, slide_pred)
        del metric_result['cm']
        result_table = PrettyTable()
        result_table.field_names = metric_result.keys()
        result_table.add_row(metric_result.values())
        logger.info('\n' + str(result_table))
        report = classification_report(slide_gt, slide_pred, target_names=["Neg", "Pos"])
        logger.info('\n' + report)
        
    torch.save(model.state_dict(), f'{files_save_dir}/checkpoints/slide_cls_best.pth')

parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=20, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(args.device)
    d_cfg = Config.fromfile(args.dataset_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    model = SlideClsNet(num_classes=2, device=device)
    model.to(device)
    train_net(cfg)


'''
python main4slide_cls_net.py \
    configs/dataset/slide_cls_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/slide_token_cls
'''