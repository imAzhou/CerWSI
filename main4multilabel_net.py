import torch
from tqdm import tqdm
from collections import OrderedDict
import argparse
from mmengine.config import Config
from cerwsi.nets import PatchMultiClsNet
from cerwsi.utils import set_seed, load_multilabel_dataset,get_logger,get_train_strategy,build_evaluator

def train_net(cfg):
    model.train()

    trainloader,valloader = load_multilabel_dataset(d_config = cfg, seed = args.seed)
    optimizer,lr_scheduler = get_train_strategy(model, cfg)
    evaluator = build_evaluator(cfg.val_evaluator)

    logger, files_save_dir = get_logger(args.record_save_dir, model, cfg, 'multi_label')
    
    for epoch in range(cfg.max_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(total=len(trainloader)*cfg.train_bs, desc=f'Train Epoch {epoch + 1}/{cfg.max_epochs}, LR: {current_lr:.6f}')
        for idx, data_batch in enumerate(trainloader):
            loss = model.train_step(data_batch, optim_wrapper=optimizer)
            postfix = OrderedDict()
            postfix['loss'] = loss.item()
            pbar.set_postfix(postfix)
            pbar.update(cfg.train_bs)

            if idx % 50 == 0:
                logger.info(f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}, loss: {loss.item():.6f}')
        pbar.close()
        lr_scheduler.step()

        model.eval()
        logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        pbar = tqdm(total=len(valloader)*cfg.val_bs, desc=f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        for idx, data_batch in enumerate(valloader):
            with torch.no_grad():
                outputs = model.val_step(data_batch)
            evaluator.process(data_samples=outputs, data_batch=data_batch)
            pbar.update(cfg.val_bs)

        metrics = evaluator.evaluate(len(valloader)*cfg.val_bs)            
        pbar.close()
        logger.info(metrics)

    torch.save(model.state_dict(), f'checkpoints/vlaid_cls_best.pth')


parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:1')

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(args.device)
    d_cfg = Config.fromfile(args.dataset_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    model = PatchMultiClsNet(num_classes = d_cfg['num_classes'], device=device)
    model.load_backbone('checkpoints/dinov2_vits14_pretrain.pth', frozen=False)
    
    train_net(cfg)

'''
python main4multilabel_net.py \
    configs/dataset/multi_label_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/debug
'''