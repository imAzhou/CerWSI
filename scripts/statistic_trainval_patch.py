import json
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

NEGATIVE_CLASS = ['NILM', 'GEC']
ASC_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
AGC_CLASS = ['AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']

def draw_hist(area_data, save_path):
    # 定义直方图的分段节点
    split_node = [0, 64, 128, 224, 448, 10000]
    bins = [i*i for i in split_node]
    axis_label = [f'{split_node[i]}² - {split_node[i+1]}²' for i in range(len(split_node)-2)]
    axis_label.append('>448²')

    df_data = []
    for area_type,area_value in area_data.items():
        counts, bin_edges = np.histogram(area_value, bins=bins)
        for idx,bin_key in enumerate(axis_label):
            df_data.append([area_type, bin_key, counts[idx]])
    
    df = pd.DataFrame(df_data, columns=["area_type", "bin_name", "bin_count"])
    g = sns.catplot(
        data=df, kind="bar",
        x="bin_name", y="bin_count", hue="area_type",
        height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "count")
    g.legend.set_title("")
    
    # 自定义 legend 标签
    new_labels = [f'{area_type}({len(area_value)})' for area_type,area_value in area_data.items()]
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    # 获取每个条形的高度（y值），并在条形上显示数值
    for ax in g.axes.flat:
        for p in ax.patches[:-len(g.legend.texts)]:
            # 获取条形的高度
            height = p.get_height()
            # 在条形顶部显示数值
            ax.text(p.get_x() + p.get_width() / 2, height, f'{int(height)}', 
                    ha='center', va='bottom', fontsize=10)
        
    plt.savefig(save_path)

def get_result(valid_imgs, mode):
    slide_clsname_cnt = dict()
    patch_clsname_cnt = dict()
    patch_clsname_areas = dict()
    patch_clsname3_areas = dict()

    for slide_valid_item in tqdm(valid_imgs, ncols=80):
        kfb_clsname = slide_valid_item['kfb_clsname']
        slide_clsname_cnt[kfb_clsname] = slide_clsname_cnt.get(kfb_clsname, 0) + 1
        for img_item in slide_valid_item['valid_anno']:
            patch_clsname = img_item['patch_clsname']
            patch_clsname_cnt[patch_clsname] = patch_clsname_cnt.get(patch_clsname, 0) + 1
            w,h = img_item['size']
            patch_clsname_areas.setdefault(patch_clsname, []).append(w * h)

            clsnamein3 = ''
            if patch_clsname in NEGATIVE_CLASS:
                clsnamein3 = 'negative'
            elif patch_clsname in ASC_CLASS:
                clsnamein3 = 'ASC'
            elif patch_clsname in AGC_CLASS:
                clsnamein3 = 'AGC'
            patch_clsname3_areas.setdefault(clsnamein3, []).append(w * h)

    result_table = PrettyTable(title=f'{mode} Slide Nums')
    result_table.field_names = ["类别"] + list(slide_clsname_cnt.keys())
    result_table.add_row(['num'] + list(slide_clsname_cnt.values()))
    print(result_table)

    custom_order = [*NEGATIVE_CLASS, *ASC_CLASS, *AGC_CLASS]
    sorted_keys = [key for key in custom_order if key in patch_clsname_cnt]
    sorted_values = [patch_clsname_cnt[key] for key in sorted_keys]
    result_table = PrettyTable(title=f'{mode} Patch Nums')
    result_table.field_names = ["类别"] + sorted_keys
    result_table.add_row(['num'] + sorted_values)
    print(result_table)

    draw_hist(patch_clsname_areas, f'data_resource/cls_pn/statistic_results/area_dist_{mode}.png')
    draw_hist(patch_clsname3_areas, f'data_resource/cls_pn/statistic_results/area_dist_in3_{mode}.png')


def statistic_anno_json():
    with open('data_resource/cls_pn/1117_anno_train.json','r') as f:
        train_data = json.load(f)
    with open('data_resource/cls_pn/1117_anno_val.json','r') as f:
        val_data = json.load(f)
    
    # 统计项目：各类别数量（11 和 3 分类），面积分布
    get_result(train_data['valid_imgs'], 'train')
    get_result(val_data['valid_imgs'], 'val')
    get_result([
        *train_data['valid_imgs'],
        *val_data['valid_imgs'],
    ], 'total')

if __name__ == '__main__':
    statistic_anno_json()