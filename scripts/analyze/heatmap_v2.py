import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from cerwsi.datasets import TokenClsDataset
import argparse
import math
from PIL import Image
from mmengine.dataset import DefaultSampler
from mmengine.config import Config
from cerwsi.nets import CerMCNet
from cerwsi.utils import set_seed
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()


def draw_heatmap(bs_attn_map, bs_image_paths, bs_gt_labels, bs_pred_labels, save_path, clsid):
    min_v,max_v = torch.min(bs_attn_map),torch.max(bs_attn_map)
    bs = len(bs_image_paths)
    grid_size = 224
    axis_size = int(math.sqrt(bs))
    fig, axes = plt.subplots(axis_size, axis_size, figsize=(axis_size*4, axis_size*4))

    for bidx, attn_map, imgpath in zip(range(bs), bs_attn_map, bs_image_paths):
        image = Image.open(imgpath)
        purename = os.path.basename(imgpath).split('.')[0]
        feat_size = int(math.sqrt(len(attn_map)))
        attn_map_2d = attn_map.reshape(feat_size, feat_size).detach().cpu().numpy()
        scale_factor = grid_size // attn_map_2d.shape[-1]
        ax = axes[bidx // axis_size, bidx % axis_size]  # 计算对应的行和列

        image_resized = image.resize((grid_size, grid_size))
        ax.imshow(image_resized)  # 显示原图

        expanded_map = zoom(attn_map_2d, (scale_factor, scale_factor), order=1)  # 双线性插值
        ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v)  # 使用热力图显示
        ax.axis('off')  # 关闭坐标轴
        gt = ','.join(map(str, bs_gt_labels[bidx]))
        pred = ','.join(map(str, bs_pred_labels[bidx]))
        color = 'red' if clsid in bs_pred_labels[bidx] else 'black'
        ax.set_title(f'{purename}: {gt}/{pred}', color = color)  # 设置标题，可以自定义
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

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
    train_sampler = DefaultSampler(train_dataset, shuffle=True, seed=args.seed),

    train_loader = DataLoader(train_dataset, 
                            pin_memory=True,
                            batch_size=cfg.train_bs, 
                            sampler = train_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    val_dataset = TokenClsDataset(cfg.data_root, 'val')
    val_sampler = DefaultSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            pin_memory=True,
                            batch_size=cfg.val_bs, 
                            sampler = val_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    
    return train_loader, val_loader

def test_net(cfg, model):
    trainloader,valloader = load_data(cfg)
    model.eval()
    for idx, data_batch in enumerate(tqdm(valloader, ncols=80)):
        if idx > 10:
            break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        bs_image_paths = outputs['image_paths']
        bs_gt_labels = [list(set([ii[-1] for ii in i])) if len(i)>0 else [0] for i in outputs['token_labels']]
        bs_pred_labels = []
        for confidence_pred in (outputs['pos_probs'] > 0.5).int():
            pred_labels = torch.nonzero(confidence_pred == 1)
            if len(pred_labels) == 0:
                bs_pred_labels.append([0])
            else:
                pred_labels = pred_labels + 1
                bs_pred_labels.append(pred_labels.squeeze(-1).tolist())

        bs, layers, num_classes, num_tokens = outputs['attn_array'].shape
        
        for clsid in range(num_classes):
            attnmap_save_dir = f'{args.save_dir}/attn_cls_visual/{CLASSES[clsid]}'
            os.makedirs(attnmap_save_dir, exist_ok=True)
            for layer_idx in range(layers):
                bs_attn_map = outputs['attn_array'][:, layer_idx, clsid, :]  # (bs, num_tokens)
                save_path = f'{attnmap_save_dir}/bs{idx}_layer{layer_idx}.png'
                draw_heatmap(bs_attn_map, bs_image_paths, bs_gt_labels, bs_pred_labels, save_path, clsid)



def main():
    set_seed(args.seed)
    device = torch.device(f'cuda:0')
    cfg = Config.fromfile(args.config_file)
    cfg.val_bs = 64
    model = CerMCNet(
        num_classes = cfg['num_classes'], 
        backbone_type = cfg.backbone_type,
        use_lora=cfg.use_lora
    ).to(device)
    model.load_ckpt(args.ckpt)
    test_net(cfg, model)


if __name__ == '__main__':
    CLASSES = ['Negative', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'AGC']
    main()

'''
python scripts/analyze/heatmap_v2.py \
    log/multi_patch_ours/2025_02_22_20_57_02/config.py \
    log/multi_patch_ours/2025_02_22_20_57_02/checkpoints/best.pth \
    log/multi_patch_ours/2025_02_22_20_57_02
'''
