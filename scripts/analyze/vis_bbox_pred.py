import matplotlib
import torch
import os
from torchvision.ops import nms
from tqdm import tqdm
import torch.distributed as dist
import math
import argparse
from mmengine.config import Config
from cerwsi.datasets import load_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

POSITIVE_THR = 0.5
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLS_COLORS = {
    'AGC': '#1f77b4',   # 蓝色
    'ASC-US': '#ff9999',  # 浅红
    'LSIL': '#ff6666',  # 中浅红
    'ASC-H': '#ff3333',  # 中红
    'HSIL': '#cc0000',  # 深红
}

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def draw_vis(img,bboxes_coords, bboxes_clsname, pred_bboxes, binary_attnmap, output_path):
    h,w = img.size
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.01)
    # ========== 子图 1：原图 + Bounding Boxes ==========
    axes[0].imshow(img)
    classname_denote = []
    for bbox, cls_name in zip(bboxes_coords, bboxes_clsname):
        x1, y1, x2, y2 = bbox
        color = CLS_COLORS.get(cls_name, 'white')
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, cls_name, color=color, fontsize=8)
        if cls_name in POSITIVE_CLASS:
            classname_denote.append(POSITIVE_CLASS.index(cls_name)+1)
    axes[0].set_title(f"gt Classes: {list(set(classname_denote))}")
    axes[0].axis("off")

    # ========== 子图 2：Token Classes 可视化 ==========
    axes[1].imshow(img)
    overlay = np.zeros((h, w, 4))  # RGBA
    classname_denote = []
    for pred_bbox in pred_bboxes:
        x1, y1, x2, y2 = pred_bbox['bbox']
        cls_name = POSITIVE_CLASS[pred_bbox['cls'] - 1]
        color = CLS_COLORS.get(cls_name, 'white')
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, cls_name, color=color, fontsize=8)
        classname_denote.append(pred_bbox['cls'])

        mask_color = np.array(matplotlib.colors.to_rgba(color, alpha=0.8))
        bbox_mask = pred_bbox['mask'].int().detach().cpu().numpy()
        overlay[bbox_mask] = mask_color  # 叠加颜色
        
    axes[1].imshow(overlay)
    axes[1].set_title(f"pred Classes: {list(set(classname_denote))}")
    axes[1].axis("off")

    # ========== 子图 3：Token Probabilities 热力图 ==========
    axes[2].imshow(img)
    norm_attnmap = cv2.normalize(binary_attnmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_attnmap = norm_attnmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_attnmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    axes[2].imshow(heatmap, alpha=0.6)
    axes[2].set_title("Binary AttnMap")
    axes[2].axis("off")

    patches = [
        mpatches.Patch(color=matplotlib.colors.to_rgba(color), label=category)  # Matplotlib 支持归一化颜色
        for category, color in CLS_COLORS.items()
    ]
    # 获取图形的尺寸（单位：英寸）
    fig_width, fig_height = fig.get_size_inches()
    # 将图例偏移图像尺寸的5%
    offset_in_inches = 4.5 / fig_width
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1+offset_in_inches, 1), frameon=False)
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def test_net(cfg, model, model_without_ddp):
    trainloader,valloader = load_data(cfg)
    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    for idx, data_batch in enumerate(pbar):
        # if idx > 20:
        #     break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        
        for bidx in range(len(outputs['images'])):
            prefix = outputs['metainfo'][bidx]['prefix']
            os.makedirs(f'{vis_save_dir}/{prefix}', exist_ok=True)
            # if random.random() < 0.1:
            if prefix == 'total_pos':
                metainfo = outputs['metainfo'][bidx]
                # bboxes_coords: [x1,y1,x2,y2], bboxes_clsname one of POSITIVE_CLASS
                img_path,bboxes_coords,bboxes_clsname = metainfo['imgpath'],metainfo['bboxes'],metainfo['clsnames']
                img = Image.open(img_path)
                h,w = img.size
                filename = os.path.basename(img_path)

                pred_bboxes = outputs['pred_bbox'][bidx]
                binary_attnmap = outputs['binary_attnmap'][bidx]
                feat_size = int(math.sqrt(binary_attnmap.shape[0]))
                binary_attnmap = binary_attnmap.reshape((feat_size, feat_size)).detach().cpu().numpy()
                binary_attnmap = cv2.resize(binary_attnmap, (w, h), interpolation=cv2.INTER_NEAREST)
                
                output_path = f'{vis_save_dir}/{prefix}/{filename}'
                draw_vis(img,bboxes_coords, bboxes_clsname, pred_bboxes, binary_attnmap, output_path)

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    model = PatchClsNet(cfg).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_without_ddp.load_ckpt(args.ckpt)
    test_net(cfg, model, model_without_ddp)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':

    vis_save_dir = 'statistic_results/WSI_heatmap_partial'
    os.makedirs(vis_save_dir, exist_ok=True)
    main()

'''
CUDA_VISIBLE_DEVICES=5,7 torchrun  --nproc_per_node=2 --master_port=12346 scripts/analyze/vis_bbox_pred.py \
    log/l_cerscanv4/wscer_partial_pure/2025_04_29_17_37_52/config.py \
    log/l_cerscanv4/wscer_partial_pure/2025_04_29_17_37_52/checkpoints/best.pth
'''