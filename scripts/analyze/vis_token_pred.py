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

def draw_vis(img,bboxes_coords, bboxes_clsname,token_classes_resized,token_probs_resized,output_path):
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
    axes[0].set_title(f"Image with partial gt: {list(set(classname_denote))}")
    axes[0].axis("off")

    # ========== 子图 2：Token Classes 可视化 ==========
    axes[1].imshow(img)
    overlay = np.zeros((h, w, 4))  # RGBA
    classname_denote = []
    total_bboxes = []
    for class_id in range(1, len(POSITIVE_CLASS) + 1):  # 只处理 > 0 的类别
        mask = (token_classes_resized == class_id)
        if np.sum(mask) > 0:
            classname_denote.append(class_id)
            cls_name = POSITIVE_CLASS[class_id - 1]
            # 获取连通区域的外接矩形框
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            bboxes = []
            scores = []
            for i in range(1, num_labels):  # 跳过背景
                x, y, bwidth, bheight, _ = stats[i]
                x1, y1, x2, y2 = x, y, x + bwidth, y + bheight
                if bwidth < 50 and bheight < 50:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    x1, y1 = center_x-25,center_y-25
                    x2, y2 = center_x+25,center_y+25
                
                # 提取第 i 个连通区域的 mask（按 label）
                region_mask = (labels == i)
                region_probs = token_probs_resized[region_mask]
                max_score = float(np.max(region_probs)) if region_probs.size > 0 else 0.0
                bboxes.append([x1, y1, x2, y2])
                scores.append(max_score)

            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.7)
            kept_boxes = boxes_tensor[keep].numpy()
            kept_boxes_score = scores_tensor[keep].numpy()
            
            for coord,score in zip(kept_boxes, kept_boxes_score):
                total_bboxes.append({
                    'coord':coord,
                    'score':score,
                    'clsname':cls_name,
                })

    if len(total_bboxes) > 20:
        class_order = POSITIVE_CLASS[::-1]  # ['HSIL', 'ASC-H', 'LSIL', 'ASC-US', 'AGC']
        # 排序：先按 clsname 在 class_order 中的索引，再按 score 降序
        total_bboxes_sorted = sorted(
            total_bboxes,
            key=lambda x: (class_order.index(x['clsname']), -x['score'])
        )
        top_bbox = total_bboxes_sorted[0]
        total_bboxes = [{
            'coord':[0,0,w,h],
            'score':top_bbox['score'],
            'clsname':top_bbox['clsname'],
        }]
    
    for boxitem in total_bboxes:
        x1, y1, x2, y2 = boxitem['coord']
        width, height = x2 - x1, y2 - y1
        color = np.array(matplotlib.colors.to_rgba(CLS_COLORS[boxitem['clsname']], alpha=0.8))
        # overlay[mask] = color  # 叠加颜色
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color[:3], facecolor='none')
        axes[1].add_patch(rect)
        
    # axes[1].imshow(overlay)
    axes[1].set_title(f"Token Classes: {classname_denote}")
    axes[1].axis("off")

    # ========== 子图 3：Token Probabilities 热力图 ==========
    axes[2].imshow(img)
    masked_probs = np.zeros_like(token_probs_resized)
    masked_probs[token_classes_resized > 0] = token_probs_resized[token_classes_resized > 0]  # 仅显示类别大于0的区域
    heatmap = cv2.applyColorMap((masked_probs * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 归一化并应用 colormap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换颜色通道
    axes[2].imshow(heatmap, alpha=0.6)
    axes[2].set_title("Token Probabilities")
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
        if idx > 1:
            break
        
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        
        for bidx in range(len(outputs['images'])):
            if random.random() < 0.1:
                token_probs,token_classes = outputs['token_probs'][bidx],outputs['token_classes'][bidx]
                feat_size = int(math.sqrt(token_probs.shape[0]))
                metainfo = outputs['metainfo'][bidx]
                # bboxes_coords: [x1,y1,x2,y2], bboxes_clsname one of POSITIVE_CLASS
                img_path,bboxes_coords,bboxes_clsname = metainfo['imgpath'],metainfo['bboxes'],metainfo['clsnames']
                img = Image.open(img_path)
                h,w = img.size
                filename = os.path.basename(img_path)
                token_classes = token_classes.reshape((feat_size, feat_size)).detach().cpu().numpy()
                token_classes_resized = cv2.resize(token_classes, (w, h), interpolation=cv2.INTER_NEAREST)
                token_probs = token_probs.reshape((feat_size, feat_size)).detach().cpu().numpy()
                token_probs_resized = cv2.resize(token_probs, (w, h), interpolation=cv2.INTER_LINEAR)
                
                output_path = f'{vis_save_dir}/{filename}'
                draw_vis(img,bboxes_coords, bboxes_clsname,token_classes_resized,token_probs_resized,output_path)

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
    test_net(cfg, model, model_without_ddp)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':

    vis_save_dir = 'statistic_results/WSI_heatmap_partial'
    os.makedirs(vis_save_dir, exist_ok=True)
    main()

'''
CUDA_VISIBLE_DEVICES=0,1,2 torchrun  --nproc_per_node=3 --master_port=12346 scripts/analyze/vis_token_pred.py \
    log/l_cerscan_v3/wscer_partial/config.py \
    log/l_cerscan_v3/wscer_partial/checkpoints/best.pth
'''