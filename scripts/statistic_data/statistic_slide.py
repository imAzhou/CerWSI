import pandas as pd
import json
from cerwsi.utils import read_json_anno
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 类别映射关系
RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC-N',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

# sorted_clsname = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC-N', 'AGC']
sorted_clsname = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']

def static_class():
    valid_info = dict()
    with open('data_resource/cls_pn/1117_anno_train.json','r') as f:
        train_data = json.load(f)
    with open('data_resource/cls_pn/1117_anno_val.json','r') as f:
        val_data = json.load(f)

    for slide_valid_item in tqdm(val_data['valid_imgs'], ncols=80):
        patientId = slide_valid_item['patientId']
        if 'JFSW' not in patientId:
            continue
        valid_info.setdefault(slide_valid_item['kfb_clsname'], []).append(slide_valid_item['valid_anno'])

    data = {}
    for key in sorted_clsname:
        data[key] = {
            'slide_num': 0,
            'clsname_cnt': {}
        }
        for subkey in sorted_clsname:
            data[key]['clsname_cnt'][subkey] = 0
    
    for slide_clsname, valid_bboxes in valid_info.items():
        data[slide_clsname]['slide_num'] = len(valid_bboxes)
        for s_valid in valid_bboxes:
            for v_bbox in s_valid:
                p_clsname = RECORD_CLASS[v_bbox['patch_clsname']]
                data[slide_clsname]['clsname_cnt'][p_clsname] += 1

    category_names = []
    annot_counts_per_category = []
    for parent in sorted_clsname:
        category_names.append(f"({data[parent]['slide_num']}) {parent}")
        
        # 统计该父类别下每个子类别的标注框数量
        counts = [data[parent]['clsname_cnt'][sub] for sub in data[parent]['clsname_cnt'].keys()]
        annot_counts_per_category.append(counts)

    # 转置数据以便绘图
    annot_counts_per_category = np.array(annot_counts_per_category).T

    # 堆叠柱状图绘制
    x = np.arange(len(category_names))  # 横轴位置
    width = 0.6  # 柱状图宽度
    colors = plt.cm.tab10(range(len(sorted_clsname)))  # 使用离散颜色方案

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, subcategory in enumerate(sorted_clsname):
        bottom = np.sum(annot_counts_per_category[:i], axis=0) if i > 0 else None
        ax.bar(x, annot_counts_per_category[i], width, label=subcategory, bottom=bottom, color=colors[i])

    # 设置横轴
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45, ha="right")

    # 图例
    ax.legend(title="clsname", loc="upper right", bbox_to_anchor=(1.25, 1))

    # 保存图像
    plt.tight_layout()
    plt.savefig("statistic_results/slide_innerclsname_dist_val.png")


def static_patch_cnt():
    valid_info, total_info = dict(), dict()
    for subkey in sorted_clsname:
        valid_info[subkey] = 0
        total_info[subkey] = 0

    df_jfsw_1 = pd.read_csv('data_resource/cls_pn/jfsw_ann/JFSW_1_kfb_info.csv')
    df_jfsw_2 = pd.read_csv('data_resource/cls_pn/jfsw_ann/JFSW_2_kfb_info.csv')
    dataframes = [df_jfsw_1, df_jfsw_2]
    merged_df = pd.concat(dataframes, ignore_index=True)

    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        if row.kfb_clsname not in sorted_clsname:
            continue
        valid_info[row.kfb_clsname] += row.valid_img_cnt
        annos = read_json_anno(row.json_path)
        for idx,i in enumerate(annos):
            region = i.get('region')
            sub_class = i.get('sub_class')
            w,h = region['width'],region['height']
            if w > 0 and h > 0 and sub_class in RECORD_CLASS.keys():
                total_info[row.kfb_clsname] += 1

    valid_cnt = [valid_info[kfb_clsname] for kfb_clsname in sorted_clsname]
    invalid_cnt = [total_info[kfb_clsname] - valid_info[kfb_clsname] for kfb_clsname in sorted_clsname]
    plt.figure(figsize=(8, 6))
    line1, = plt.plot(sorted_clsname, valid_cnt, marker='o', label='valid', color='blue')
    line2, = plt.plot(sorted_clsname, invalid_cnt, marker='o', label='invalid', color='orange')

    plt.legend(handles=[line1, line2], loc='upper right',)

    # 显示图形
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格以提高可读性
    plt.savefig('statistic_results/slide_patch_cnt.png')


if __name__ == '__main__':
    # static_class()
    static_patch_cnt()
