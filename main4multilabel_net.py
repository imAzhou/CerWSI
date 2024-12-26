import torch
import os
from tqdm import tqdm
from collections import OrderedDict
import torch.distributed as dist
import argparse
from mmengine.config import Config
from cerwsi.nets import PatchMultiClsNet
from cerwsi.utils import set_seed, init_distributed_mode, load_multilabel_dataset, get_logger, get_train_strategy, build_evaluator,reduce_loss,is_main_process


# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def train_net(cfg, model, model_without_ddp, device):
    trainloader,valloader = load_multilabel_dataset(
        d_config = cfg, seed = args.seed, distributed = args.distributed)
    optimizer,lr_scheduler = get_train_strategy(model_without_ddp, cfg)
    evaluator = build_evaluator(cfg.val_evaluator)
    
    if is_main_process():
        logger, files_save_dir = get_logger(args.record_save_dir, model_without_ddp, cfg, 'multi_label')
    max_acc = -1
    for epoch in range(cfg.max_epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = trainloader
        if is_main_process():
            pbar = tqdm(trainloader, ncols=80)
        
        avg_loss = torch.zeros(1, device=device)
        for idx, data_batch in enumerate(pbar):
            loss = model(data_batch, 'train', optim_wrapper=optimizer)
            loss = reduce_loss(loss)
            avg_loss = (avg_loss * idx + loss.detach()) / (idx + 1)
            if is_main_process():
                pbar.desc = f"average loss: {round(avg_loss.item(), 4)}"

            if idx % 50 == 0 and is_main_process():
                logger.info(f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}, average loss: {avg_loss.item():.6f}')
        if is_main_process():
            pbar.close()
        lr_scheduler.step()
        
        if (epoch+1) % 10 == 0:
            model.eval()
            pbar = valloader
            if is_main_process():
                logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
                pbar = tqdm(valloader, ncols=80)
            
            for idx, data_batch in enumerate(pbar):
                with torch.no_grad():
                    outputs = model(data_batch, 'val')
                evaluator.process(data_samples=outputs, data_batch=data_batch)

            metrics = evaluator.evaluate(len(valloader.dataset))
            if is_main_process():
                pbar.close()
                logger.info(metrics)
                if metrics['multi-label/f1-score'] > max_acc:
                    max_acc = metrics['multi-label/f1-score']
                    torch.save(model_without_ddp.state_dict(), f'{files_save_dir}/checkpoints/best.pth')


def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    d_cfg = Config.fromfile(args.dataset_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    model = PatchMultiClsNet(num_classes = d_cfg['num_classes'], device=device).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_without_ddp.load_backbone('checkpoints/dinov2_vits14_pretrain.pth', frozen=False)
    train_net(cfg, model, model_without_ddp, device)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12345 main4multilabel_net.py \
    configs/dataset/multi_label_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/multilabel
'''