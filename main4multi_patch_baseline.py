import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from cerwsi.datasets import TokenClsDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
from mmengine.config import Config
from cerwsi.nets import MultiResNet,MultiVit
from cerwsi.utils import MultiPosMetric
from cerwsi.utils import set_seed, init_distributed_mode, get_logger, get_train_strategy, build_evaluator,reduce_loss,is_main_process

POSITIVE_THR = 0.5
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def load_data(cfg):
    def custom_collate(batch):
        # 拆分 batch 中的图像和标签
        images = [item[0] for item in batch]  # 所有 image_tensor，假设 shape 一致
        image_labels = [item[1] for item in batch]
        token_labels = [item[2] for item in batch]

        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)

        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'token_labels': token_labels  # 保持 label 的原始列表形式
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


def train_net(cfg, model, model_without_ddp):
    trainloader,valloader = load_data(cfg)
    optimizer,lr_scheduler = get_train_strategy(model_without_ddp, cfg)
    
    # evaluator = build_evaluator(cfg.val_evaluator)
    evaluator = build_evaluator([MultiPosMetric(thr=POSITIVE_THR)])
    
    if is_main_process():
        logger, files_save_dir = get_logger(args.record_save_dir, model_without_ddp, cfg, 'multi_token')
    max_acc = -1
    for epoch in range(cfg.max_epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = trainloader
        if is_main_process():
            pbar = tqdm(trainloader, ncols=80)
        
        # avg_loss = torch.zeros(1, device=device)
        for idx, data_batch in enumerate(pbar):
            # if idx > 10:
            #     break
            loss = model(data_batch, 'train', optim_wrapper=optimizer)
            loss = reduce_loss(loss)
            # avg_loss = (avg_loss * idx + loss.detach()) / (idx + 1)
            if is_main_process():
                pbar.desc = f"average loss: {round(loss.item(), 4)}"

            if idx % 50 == 0 and is_main_process():
                logger.info(f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}, average loss: {loss.item():.6f}')
        if is_main_process():
            pbar.close()
        lr_scheduler.step()
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            pbar = valloader
            if is_main_process():
                logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
                pbar = tqdm(valloader, ncols=80)
            
            for idx, data_batch in enumerate(pbar):
                # if idx > 10:
                #     break
                with torch.no_grad():
                    outputs = model(data_batch, 'val')
                evaluator.process(data_samples=[outputs], data_batch=None)

            metrics = evaluator.evaluate(len(valloader.dataset))
            if is_main_process():
                pbar.close()
                logger.info(metrics)
                prime_metric = 'multi-label/img_accuracy'
                if metrics[prime_metric] > max_acc:
                    max_acc = metrics[prime_metric]
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
    
    # model = MultiResNet(num_classes = d_cfg['num_classes']).to(device)
    model = MultiVit(num_classes = d_cfg['num_classes'], backbone_type=cfg.baseline_backbone).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # model_without_ddp.load_backbone('checkpoints/resnet50_a1_0-14fe96d1.pth',frozen=False)
    train_net(cfg, model, model_without_ddp)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node=2 --master_port=12345 main4multi_patch_baseline.py \
    configs/dataset/multi_patch_uni_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/multi_patch_baseline
'''