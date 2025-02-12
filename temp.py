from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 假设 y_true 是真实标签，y_scores 是模型预测的概率
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3])
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(thresholds)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('temp.png')


import torch, torchvision
import torch.nn as nn
from cerwsi.nets.CHIEF import CHIEF


model = CHIEF(size_arg="small", dropout=True, n_classes=2)

model.eval()
