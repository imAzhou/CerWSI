import pickle
import json
import os
from tqdm import tqdm
from prettytable import PrettyTable
from mmpretrain.evaluation import MultiLabelMetric
from cerwsi.utils import calculate_metrics


def analyze():
    with open(val_json_path, 'r') as f:
        gtinfo = json.load((f))
    with open(pkl_path, 'rb') as f:
        pred_result = pickle.load(f)
    
    img2gt = {}
    for imggt in tqdm(gtinfo, ncols=80):
        filename = imggt['filename']
        gt_multi_label = list(set([tk[-1] for tk in imggt['gtmap_14']]))
        if len(gt_multi_label) == 0:
            gt_multi_label = [0]
        
        img2gt[filename] = dict(
            multi_label = gt_multi_label,
            image_label = imggt['diagnose'],
        )
    
    total_gt_img_label,total_pred_img_label = [],[]
    total_gt_multi_label,total_pred_multi_label = [],[]
    for imgpred in tqdm(pred_result, ncols=80):
        filename = os.path.basename(imgpred['img_path'])
        total_gt_multi_label.append(img2gt[filename]['multi_label'])
        total_gt_img_label.append(img2gt[filename]['image_label'])
        
        pred_mlabel = imgpred['pred_instances']['labels']
        pred_mlabel = [i+1 for i in pred_mlabel]    # 目标检测模型预测的类别 id 从 0 开始，与多标签分类模型的类别 id 对齐需要加 1
        pred_image_label = 1
        if len(pred_mlabel) == 0:
            pred_mlabel = [0]
            pred_image_label = 0
        total_pred_multi_label.append(pred_mlabel)
        total_pred_img_label.append(pred_image_label)
    
    multi_result = MultiLabelMetric.calculate(
        total_pred_multi_label,
        total_gt_multi_label,
        pred_indices=True,
        target_indices=True,
        average=None,
        num_classes=num_classes)
    precision, recall, f1_score, support = multi_result
    img_result = calculate_metrics(total_gt_img_label, total_pred_img_label)
    del img_result['cm']
    print(f'img_result: {img_result}')
    
    _precision = [round(i,2) for i in precision.detach().cpu().tolist()]
    _recall = [round(i,2) for i in recall.detach().cpu().tolist()]
    _f1_score = [round(i,2) for i in f1_score.detach().cpu().tolist()]
    result_table = PrettyTable()
    result_table.field_names = ['Metric', 'NILM', 'ASC-US','LSIL', 'ASC-H','HSIL', 'AGC']
    result_table.add_row(['Precision', *_precision])
    result_table.add_row(['Recall', *_recall])
    result_table.add_row(['F1', *_f1_score])
    print(result_table)

if __name__ == '__main__':
    pkl_root_dir = '/c22073/zly/codes/mmdetection/work_dirs'
    pkl_path = f'{pkl_root_dir}/faster_rcnn_r50/pred_result.pkl'
    json_root_dir = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    val_json_path = f'{json_root_dir}/annofiles/val_patches.json'

    num_classes = 6

    analyze()
