import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def collect_confi():
    df_val = pd.read_csv('data_resource/cls_pn/1117_val.csv')
    confi_txt_dir = f'{root_dir}/posi_conf'
    patch_clsname_confis = dict()
    for txt_filename in tqdm(os.listdir(confi_txt_dir),ncols=80):
        with open(f'{confi_txt_dir}/{txt_filename}', 'r') as txtf:
            lines = txtf.readlines()
        patientId = txt_filename.split('.')[0]
        filtered = df_val.loc[df_val['patientId'] == patientId]
        if filtered.empty:
            print("No matching rows found.")
            continue
        kfb_info = filtered.iloc[0]
        if kfb_info.kfb_clsid == 0:
            continue
        confis = [float(line.strip()) for line in lines if float(line.strip()) > 0.3]
        patch_clsname_confis.setdefault(kfb_info.kfb_clsname, []).append(confis)
    
    return patch_clsname_confis


def draw_hist(data):
    # 示例数据
    # data = {
    #     "ASC-US": [[0.1, 0.2, 0.35, 0.8, 0.65]],
    #     "LSIL": [[0.5, 0.6, 0.75, 0.9, 0.2]],
    #     "ASC-H": [[0.1, 0.25, 0.4, 0.6, 0.9, 0.85]],
    #     "HSIL": [[0.3, 0.4, 0.5, 0.7, 0.8]]
    # }

    # 定义置信度区间
    bins = [0.3, 0.5, 0.7, 1.0]
    bin_labels = ["[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 1.0]"]

    # 计算每个类别在各置信度区间的样本数量
    category_names = []
    counts_per_bin = []
    for category, confidences in data.items():
        category_names.append(f"({len(confidences)}){category}")
        
        counts, _ = np.histogram([y for x in confidences for y in x], bins=bins)
        counts_per_bin.append(counts)

    # 转置数据以便绘图
    counts_per_bin = np.array(counts_per_bin).T

    # 绘制堆叠柱状图
    x = np.arange(len(category_names))  # 横轴位置
    width = 0.3  # 柱状图宽度

    # 定义颜色
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    # 堆叠柱状图
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(len(bin_labels)):
        ax.bar(x, counts_per_bin[i], width, label=bin_labels[i], bottom=np.sum(counts_per_bin[:i], axis=0), color=colors[i])

    # 设置横轴
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45, ha="right")
    # ax.set_xlabel("类别 (样本数量)")
    # ax.set_ylabel("样本数量")
    # ax.set_title("各类别样本置信度分布")

    # 图例
    ax.legend(title="confi", loc="upper right")

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{root_dir}/confidence_distribution.png")

if __name__ == '__main__':
    root_dir = 'log/1117_val_origin'
    # root_dir = 'log/1117_val'
    patch_clsname_confis = collect_confi()
    draw_hist(patch_clsname_confis)