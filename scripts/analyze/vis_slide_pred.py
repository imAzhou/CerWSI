import json
from tqdm import tqdm
from cerwsi.utils import KFBSlide
import os
import matplotlib
from tqdm import tqdm
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.pyplot as plt

POSITIVE_THR = 0.5
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLS_COLORS = {
    'AGC': '#1f77b4',   # 蓝色
    'ASC-US': '#ff9999',  # 浅红
    'LSIL': '#ff6666',  # 中浅红
    'ASC-H': '#ff3333',  # 中红
    'HSIL': '#cc0000',  # 深红
}

def draw_vis(img, pos_bboxes, start_xy, output_path):
    # 创建画布
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    classname_denote = []
    px1,py1 = start_xy
    for bbox_item in pos_bboxes:
        x1, y1, x2, y2 = bbox_item['coord']
        x1, y1, x2, y2 = x1-px1, y1-py1,x2-px1, y2-py1
        cls_name = bbox_item['clsname']
        color = CLS_COLORS.get(cls_name, 'white')
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, cls_name, color=color, fontsize=8)
        if cls_name in POSITIVE_CLASS:
            classname_denote.append(POSITIVE_CLASS.index(cls_name)+1)
    ax.set_title(f"Image with pos bbox: {list(set(classname_denote))}")
    ax.axis("off")

    # patches = [
    #     mpatches.Patch(color=matplotlib.colors.to_rgba(color), label=category)  # Matplotlib 支持归一化颜色
    #     for category, color in CLS_COLORS.items()
    # ]
    # # 获取图形的尺寸（单位：英寸）
    # fig_width, fig_height = fig.get_size_inches()
    # # 将图例偏移图像尺寸的5%
    # offset_in_inches = 1.5 / fig_width
    # plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1+offset_in_inches, 1), frameon=False)
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    patientId = 'ZY_ONLINE_1_101'
    json_path = 'log/wscer_partial/pred_pos_items/ZY_ONLINE_1_101.json'
    kfb_path = '/nfs-medical/vipa-medical/zheyi/zly/KFBs/till_0318/ASC-H/YC202403898-633.kfb'
    output_dir = 'statistic_results/slide_pred_posbboxes'
    os.makedirs(output_dir, exist_ok=True)
    slide = KFBSlide(kfb_path)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    for pidx,patch_predinfo in enumerate(tqdm(json_data, ncols=80)):
        pred_label = patch_predinfo['pn_pred']['pred_label']
        if pred_label == 0:
            continue
        px1,py1,px2,py2 =  patch_predinfo['coords']
        width,height = px2-px1, py2-py1
        location, level, size = (px1,py1), 0, (width, height)
        read_result = Image.fromarray(slide.read_region(location, level, size))
        pos_bboxes = patch_predinfo['pn_pred']['pos_bboxes']
        output_path = f'{output_dir}/{patientId}_{pidx}.png'
        
        draw_vis(read_result, pos_bboxes, (px1,py1), output_path)
