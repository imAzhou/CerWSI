import argparse
from cerwsi.nets import MultiPatchUNI
from cerwsi.utils import set_seed
import torch
import os
from PIL import Image
from torchvision import transforms
from mmengine.config import Config
import math
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def draw_heatmap(attn_map, image, pred_result, scale_factor, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 遍历每个类别并绘制其注意力图
    for i in range(attn_map.shape[0]):
        ax = axes[i // 3, i % 3]  # 计算对应的行和列
        ax.imshow(image, aspect='auto')  # 显示原图
        expanded_map = zoom(attn_map[i], (scale_factor, scale_factor), order=0)  # 最近邻插值
        ax.imshow(expanded_map, cmap='gray', alpha=0.5, vmin=0, vmax=1)  # 使用灰度图显示
        ax.axis('off')  # 关闭坐标轴
        ax.set_title(classnames[i] + f': {pred_result[i]}')  # 设置标题，可以自定义
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)

def visual_heatmap():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for sub_dirname in os.listdir(sample_img_dir):
        for img_filename in os.listdir(f'{sample_img_dir}/{sub_dirname}'):
            imgpath = f'{img_root_dir}/images/Pos/{img_filename}'
            image = Image.open(imgpath)
            input_tensor = transform(image)
            databatch = {
                'images': input_tensor.unsqueeze(0)
            }

            with torch.no_grad():
                outputs = model(databatch, 'val')
            attn_map = outputs['attn_map'][0]   # (num_classes, num_tokens)
            img_pn = int(outputs['img_probs'].item() > 0.5)
            pos_pred = (outputs['pos_probs'][0] > 0.5).int().detach().tolist()
            pred_result = [img_pn, *pos_pred]

            num_classes, num_tokens = attn_map.shape
            feat_size = int(math.sqrt(num_tokens))
            attn_map_2d = attn_map.reshape(num_classes, feat_size, feat_size)
            scale_factor = image.size[0] // attn_map_2d.shape[-1]
            attnmap_save_dir = f'{args.save_dir}/attn_visual/{sub_dirname}'
            os.makedirs(attnmap_save_dir, exist_ok=True)
            save_path = f'{attnmap_save_dir}/{img_filename}'
            draw_heatmap(attn_map_2d.detach().cpu().numpy(), image, pred_result, scale_factor, save_path)


if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(f'cuda:0')
    d_cfg = Config.fromfile(args.dataset_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    model = MultiPatchUNI(
        num_classes = d_cfg['num_classes'], 
        use_lora=cfg.use_lora,
        temperature=cfg.temperature
    ).to(device)
    
    classnames = ['P/N', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    sample_img_dir = 'statistic_results/cdetector'
    img_root_dir = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    visual_heatmap()

'''
python scripts/analyze/heatmap.py \
    configs/dataset/cdetector_dataset.py \
    configs/train_strategy.py \
    log/cdetector_ours/2025_02_16_16_29_38/checkpoints/best.pth \
    log/cdetector_ours/2025_02_16_16_29_38
'''