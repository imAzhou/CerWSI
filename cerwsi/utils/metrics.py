from sklearn.metrics import accuracy_score, confusion_matrix
from prettytable import PrettyTable
import numpy as np

def calculate_metrics(y_true, y_pred):
    # 准确率 (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)

    # 特异性 (Specificity)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        'cm': cm
    }


def print_confusion_matrix(cm):
    result_table = PrettyTable(title='confusion matrix')
    row_sums = np.sum(cm, axis=1).reshape(-1, 1)
    col_sums = np.sum(cm, axis=0).reshape(1, -1)
    # 构建一个扩展的矩阵，包含混淆矩阵和行列求和
    cm_with_sums = np.vstack([np.hstack([cm, row_sums]), np.hstack([col_sums, [[np.sum(cm)]]])])

    result_table.field_names = ['','0','1','sum']
    result_table.add_row(['0'] + list(cm_with_sums[0]))
    result_table.add_row(['1'] + list(cm_with_sums[1]))
    result_table.add_row(['sum'] + list(cm_with_sums[2]))
    print(result_table)
