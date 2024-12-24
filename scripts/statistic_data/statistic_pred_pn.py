import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics,print_confusion_matrix
from sklearn.metrics import classification_report


def draw_plot():

    csv_file = f'{log_root_dir}/pred_pn.csv'
    df_csv = pd.read_csv(csv_file)
    slide_sta = [[],[]]
    for row in df_csv.itertuples(index=False):
        kfb_clsid = row.kfb_clsid
        # p_path_num,n_patch_num
        p_num, n_num = row.p_path_num, row.n_patch_num
        p_ratio = p_num / (p_num + n_num)
        slide_sta[kfb_clsid].append(p_ratio)

    y1 = slide_sta[0]
    y2 = slide_sta[1]
    width = max(len(y1), len(y2)) * 0.5
    plt.figure(figsize=(width, 6))

    plt.plot(range(1, len(y1)+1), y1, color='blue', label='Negative')
    plt.plot(range(1, len(y2)+1), y2, color='red', label='positive')
    plt.legend()
    plt.xticks(range(1, max(len(y1), len(y2)) + 1))
    plt.xlabel('Slide Idx')
    plt.ylabel('positive patch ratio')
    plt.tight_layout()
    plt.savefig(f'{log_root_dir}/positive_ratio.png')

def calc_evaluation():
    positive_ratio_thr = 0.000
    slide_pred, slide_gt = [],[]
    csv_file = f'{log_root_dir}/pred_pn.csv'
    df_csv = pd.read_csv(csv_file)
    for row in df_csv.itertuples(index=False):
        slide_gt.append(row.kfb_clsid)
        # p_path_num,n_patch_num
        p_num, n_num = row.p_path_num, row.n_patch_num
        p_ratio = p_num / (p_num + n_num)
        pred_clsid = int(p_ratio > positive_ratio_thr)
        slide_pred.append(pred_clsid)
    metric_result = calculate_metrics(slide_gt, slide_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)

    print_confusion_matrix(cm)

    report = classification_report(slide_gt, slide_pred, target_names=["Neg", "Pos"])
    print(report)




if __name__ == '__main__':
    log_root_dir = 'log/1127_val_rcp_c6_70'
    # draw_plot()
    calc_evaluation()
