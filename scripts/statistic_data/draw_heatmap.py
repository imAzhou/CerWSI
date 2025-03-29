import json
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom
from PIL import Image
import torch
import cv2
import os
import pandas as pd
from cerwsi.utils import KFBSlide
import numpy as np


def draw_WSI_heatmap(slide, pn_heatmaps, save_path):
    LEVEL = 5
    width, height = slide.level_dimensions[LEVEL]
    downsample_ratio = slide.level_downsamples[LEVEL]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    location, level, size = (0, 0), LEVEL, (width, height)
    read_result = Image.fromarray(slide.read_region(location, level, size))
    patch_size = (pn_heatmaps[0]['coords'][2] - pn_heatmaps[0]['coords'][0]) / downsample_ratio
    token_h = round(height / patch_size)
    token_w = round(width / patch_size)
    img_heatmap = np.zeros((height, width))
    for item in pn_heatmaps:
        sca_x1,sca_y1,sca_x2,sca_y2 = np.array(item['coords']) / downsample_ratio
        pos_confi = item['pn_pred'][1]
        img_heatmap[int(sca_y1):int(sca_y2), int(sca_x1):int(sca_x2)] = pos_confi
    img_heatmap_resized = cv2.resize(img_heatmap, (token_w, token_h), interpolation=cv2.INTER_NEAREST)
    img_heatmap_smooth = cv2.resize(img_heatmap_resized, (width, height), interpolation=cv2.INTER_LINEAR)

    ax.imshow(read_result)
    ax.imshow(img_heatmap_smooth, cmap='jet', alpha=0.3)
    
    # 移除边距
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def draw_patch_heatmap(slide, pn_heatmaps, save_dir):

    for idx, item in enumerate(pn_heatmaps):
        pred_clsid = item['pn_pred'][0]
        if pred_clsid == 0:
            continue
        
        ori_x1,ori_y1,ori_x2,ori_y2 = item['coords']
        patch_size = int(ori_x2 - ori_x1)
        location, level, size = (ori_x1, ori_y1), 0, (patch_size, patch_size)
        read_result = Image.fromarray(slide.read_region(location, level, size))
        filename = f'POS_{idx}.png'
        attn_array = torch.as_tensor(item['attn_array'])
        
        pos_clsid = np.argmax(item['pn_pred'][2:]) + 1
        cls_attn_array = attn_array[pos_clsid]
        feat_size = int(math.sqrt(len(cls_attn_array)))
        attn_map_2d = cls_attn_array.reshape(feat_size, feat_size).numpy()
        img_heatmap_smooth = cv2.resize(attn_map_2d, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    
        fig, ax = plt.subplots(figsize=(patch_size / 100, patch_size / 100), dpi=100)
        ax.set_xlim(0, patch_size)
        ax.set_ylim(0, patch_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        ax.imshow(read_result)
        ax.imshow(img_heatmap_smooth, cmap='jet', alpha=0.6) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'{save_dir}/{filename}', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

    # ax.imshow(, cmap='gray', origin='upper')
    # ax.imshow(read_result)
    # total_heatmap = torch.as_tensor([item['attn_array'][0] for item in pn_heatmaps])
    # total_heatmap = torch.sigmoid(total_heatmap)
    # total_heatmap = (total_heatmap - total_heatmap.min()) / (total_heatmap.max() - total_heatmap.min())
    # total_heatmap = torch.pow(total_heatmap, 0.5)   # gamma < 1 提高高值的亮度
    # colors = [(0, "black"), (0.2, "red"), (0.5, "white"), (1, "white")]
    # custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_hot", colors)
    # min_v, max_v = torch.min(total_heatmap), torch.max(total_heatmap)
    # for item in pn_heatmaps:
    #     sca_x1,sca_y1,sca_x2,sca_y2 = np.array(item['coords']) / downsample_ratio
    #     heatmap_data = torch.as_tensor(item['attn_array'][0])

    #     # 叠加热力图
    #     feat_size = int(math.sqrt(len(heatmap_data)))
    #     attn_map_2d = heatmap_data.reshape(feat_size, feat_size)
    #     scale_factor = (sca_x2 - sca_x1) // attn_map_2d.shape[-1]
    #     expanded_map = zoom(attn_map_2d, (scale_factor, scale_factor), order=1)  # 双线性插值
    #     ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v, extent=[sca_x1,sca_x2,sca_y2,sca_y1], origin='upper')
        # ax.imshow(expanded_map, cmap='hot', alpha=0.6, extent=[sca_x1,sca_x2,sca_y2,sca_y1], origin='upper')
    
def draw_patch_heatmap_v2(slide, pn_heatmaps, save_dir):
    classnames = ['P/N', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    for idx, item in enumerate(pn_heatmaps):
        pred_clsid = item['pn_pred'][0]
        if pred_clsid == 0:
            continue
        
        ori_x1,ori_y1,ori_x2,ori_y2 = item['coords']
        patch_size = int(ori_x2 - ori_x1)
        location, level, size = (ori_x1, ori_y1), 0, (patch_size, patch_size)
        read_result = Image.fromarray(slide.read_region(location, level, size))
        filename = f'POS_{idx}.png'
        attn_array = torch.sigmoid(torch.as_tensor(item['attn_array']))
        min_v,max_v = torch.min(attn_array),torch.max(attn_array)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for i in range(attn_array.shape[0]):
            ax = axes[i // 3, i % 3]  # 计算对应的行和列
            ax.imshow(read_result, aspect='auto')  # 显示原图
            cls_attn_array = attn_array[i]
            feat_size = int(math.sqrt(len(cls_attn_array)))
            attn_map_2d = cls_attn_array.reshape(feat_size, feat_size).numpy()
            scale_factor = patch_size // attn_map_2d.shape[-1]
            expanded_map = zoom(attn_map_2d, (scale_factor, scale_factor), order=1)  # 双线性插值
            ax.imshow(expanded_map, cmap='hot', alpha=0.6, vmin=min_v, vmax=max_v)  # 使用热力图显示
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(classnames[i] + f': {item["pn_pred"][1:][i]}')  # 设置标题，可以自定义
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{filename}', dpi=100)
        plt.close()


if __name__ == '__main__':
    patientId = 'JFSW_1_46'
    json_path = f'log/debug_ours/heat_value/{patientId}.json'
    img_savedir = f'log/debug_ours/heatmaps_png/{patientId}'
    os.makedirs(img_savedir, exist_ok=True)
    csv_path = 'data_resource/debug2.csv'
    df_data = pd.read_csv(csv_path)
    patient_row = df_data.loc[df_data['patientId'] == patientId].iloc[0]
    slide = KFBSlide(patient_row.kfb_path)

    with open(json_path, 'r') as f:
        heatdata = json.load(f)
    draw_WSI_heatmap(slide, heatdata, f'{img_savedir}/WSI.png')
    draw_patch_heatmap(slide, heatdata, img_savedir)
