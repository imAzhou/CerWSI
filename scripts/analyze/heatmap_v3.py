import argparse
from cerwsi.nets import CerMCNet
from cerwsi.utils import set_seed
import torch
import os
from PIL import ImageDraw
import json
from PIL import Image
from torchvision import transforms
from mmengine.config import Config
import math
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def draw_heatmap(attn_map, image, pred_result, item_inside, scale_factor, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    sq_x1,sq_y1,sq_w,sq_h = [0,0,700,700]
    draw = ImageDraw.Draw(image)
    for box_item in item_inside:
        category = box_item['clsname']
        x1, y1, x2, y2 = box_item['box_x1y1x2y2']
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = 'red'
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    
    # 遍历每个类别并绘制其注意力图
    for i in range(attn_map.shape[0]):
        ax = axes[i // 3, i % 3]  # 计算对应的行和列
        ax.imshow(image, aspect='auto')  # 显示原图
        # expanded_map = zoom(attn_map[i], (scale_factor, scale_factor), order=0)  # 最近邻插值
        # ax.imshow(expanded_map, cmap='gray', alpha=0.5, vmin=0, vmax=1)  # 使用灰度图显示
        expanded_map = zoom(attn_map[i], (scale_factor, scale_factor), order=1)  # 双线性插值
        ax.imshow(expanded_map, cmap='hot', alpha=0.6)  # 使用热力图显示
        ax.axis('off')  # 关闭坐标轴
        ax.set_title(classnames[i] + f': {pred_result[i]}')  # 设置标题，可以自定义
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def visual_heatmap():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for img_filename in os.listdir(target_dir):
        imgpath = f'{target_dir}/{img_filename}'
        image = Image.open(imgpath)
        input_tensor = transform(image)
        databatch = {
            'images': input_tensor.unsqueeze(0)
        }

        with torch.no_grad():
            outputs = model(databatch, 'val')
        attn_array = outputs['attn_array'][0]   # (3, num_classes, num_tokens)
        img_pn = int(outputs['img_probs'].item() > 0.5)
        pos_pred = (outputs['pos_probs'][0] > 0.5).int().detach().tolist()
        if sum(pos_pred) > 0:
            img_pn = 1
        pred_result = [img_pn, *pos_pred]

        item_inside = []
        for ann in img_annos:
            if img_filename == ann['filename']:
                item_inside = ann['item_inside']
                break
        num_classes, num_tokens = attn_array[0].shape
        feat_size = int(math.sqrt(num_tokens))
        attnmap_save_dir = f'{args.save_dir}/attn_array_visual'
        os.makedirs(attnmap_save_dir, exist_ok=True)
        for idx,attn_map in enumerate(attn_array):
            # attn_map = torch.sigmoid(attn_map)
            attn_map_2d = attn_map.reshape(num_classes, feat_size, feat_size)
            scale_factor = image.size[0] // attn_map_2d.shape[-1]
            purename = img_filename.split('.')[0]
            save_path = f'{attnmap_save_dir}/{purename}_{idx}.png'
            draw_heatmap(attn_map_2d.detach().cpu().numpy(), image, pred_result, item_inside, scale_factor, save_path)


if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(f'cuda:0')
    cfg = Config.fromfile(args.config_file)
    
    model = CerMCNet(
        num_classes = cfg['num_classes'], 
        backbone_type = cfg.backbone_type,
        use_lora=cfg.use_lora
    ).to(device)
    model.load_ckpt(args.ckpt)
    
    classnames = ['P/N', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    target_dir = 'statistic_results/WSI_heatmap/images'
    # json_path = 'statistic_results/WSI_heatmap/cut_list.json'
    json_path = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset/annofiles/OD_instances_val.json'
    with open(json_path, 'r') as f:
        img_annos = json.load(f)
    visual_heatmap()

'''
python scripts/analyze/heatmap_v3.py \
    log/multi_patch_ours/2025_02_22_20_57_02/config.py \
    log/multi_patch_ours/2025_02_22_20_57_02/checkpoints/best.pth \
    statistic_results/WSI_heatmap
'''