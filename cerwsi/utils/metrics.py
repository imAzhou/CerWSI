from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix,classification_report
from prettytable import PrettyTable
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation import MultiLabelMetric

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


class MyMultiLabelMetric(MultiLabelMetric):
    def __init__(self,**args) -> None:
        super(MyMultiLabelMetric, self).__init__(**args)

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        thr = self.thr if self.thr else 0.3
        self.num_classes = data_samples[0]['num_classes']
        for data_sample in data_samples:
            pred_true = data_sample['pred_score'] > thr
            if sum(pred_true[1:]) == 0:
                # data_sample['pred_score'][0] > 0.7
                pred_label = torch.as_tensor([0]).to(pred_true.device)
            else:
                pred_label = torch.where(pred_true[1:])[0] + 1
            result = dict(
                gt_label = data_sample['gt_label'],
                pred_label = pred_label
            )

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        metrics = {}
        
        pred = [rs['pred_label'] for rs in results]
        target = [rs['gt_label'] for rs in results]
        metric_res = self.calculate(
            pred,
            target,
            pred_indices=True,
            target_indices=True,
            average=self.average,
            num_classes=self.num_classes)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        if self.thr:
            suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
            for k, v in pack_results(*metric_res).items():
                metrics[k + suffix] = v
        else:
            for k, v in pack_results(*metric_res).items():
                metrics[k + f'_top{self.topk}'] = v

        result_metrics = dict()
        for k, v in metrics.items():
            if self.average is None:
                result_metrics[k + '_classwise'] = v.detach().cpu().tolist()
            elif self.average == 'macro':
                result_metrics[k] = v.item()
            else:
                result_metrics[k + f'_{self.average}'] = v.item()
        return result_metrics

class MyMultiTokenMetricV1(MultiLabelMetric):
    def __init__(self,**args) -> None:
        super(MyMultiTokenMetricV1, self).__init__(**args)

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        thr = self.thr if self.thr else 0.3
        data_samples = data_samples[0]
        bs_img_gt = data_samples['image_labels']
        bs_img_pred = (data_samples['img_probs'] > thr).int()
        bs = bs_img_gt.shape[0]
        
        cls_feat_gt = data_samples['feat_gt']   # num_classes, bs, num_tokens
        csl_feat_pred = (data_samples['feat_probs'] > thr).int()   # num_classes, bs, num_tokens

        self.num_classes = cls_feat_gt.shape[0]

        for bidx in range(bs):
            feat_gt = cls_feat_gt[:,bidx,:]
            feat_pred = csl_feat_pred[:,bidx,:]
            gt_multi_label = torch.nonzero((feat_gt > 0).any(dim=1), as_tuple=False).squeeze(1)
            pred_multi_label = torch.nonzero((feat_pred > 0).any(dim=1), as_tuple=False).squeeze(1)

            result = dict(
                img_gt = bs_img_gt[bidx],
                img_pred = bs_img_pred[bidx],
                gt_multi_label = gt_multi_label,
                pred_multi_label = pred_multi_label,
            )

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        result_metrics = dict()

        img_gt = [rs['img_gt'] for rs in results]
        img_pred = [rs['img_pred'] for rs in results]
        img_result = calculate_metrics(img_gt,img_pred)
        for k,v in img_result.items():
            if k != 'cm':
                result_metrics[k] = v

        gt_multi_label = [rs['gt_multi_label'] for rs in results]
        pred_multi_label = [rs['pred_multi_label'] for rs in results]
        metric_res = self.calculate(
            pred_multi_label,
            gt_multi_label,
            pred_indices=True,
            target_indices=True,
            average=None,
            num_classes=self.num_classes)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics
        
        suffix = '_classwise' if self.thr == 0.5 else f'_thr-{self.thr:.2f}_classwise'
        for k, v in pack_results(*metric_res).items():
            result_metrics[k + suffix] = v.detach().cpu().tolist()
        
        return result_metrics

class MyMultiTokenMetric(MultiLabelMetric):
    def __init__(self,**args) -> None:
        super(MyMultiTokenMetric, self).__init__(**args)

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        thr = self.thr if self.thr else 0.3
        data_samples = data_samples[0]
        bs_img_gt = data_samples['image_labels']
        bs_img_pred = (data_samples['img_probs'] > thr).int()
        bs = bs_img_gt.shape[0]
        
        cls_feat_gt = data_samples['feat_gt']   # bs, num_tokens
        csl_feat_pred = torch.argmax(data_samples['feat_probs'], dim=-1)   # bs, num_tokens

        self.num_classes = data_samples['feat_probs'].shape[-1]

        for bidx in range(bs):
            feat_gt = cls_feat_gt[bidx,:]
            feat_pred = csl_feat_pred[bidx,:]
            gt_multi_label = torch.unique(feat_gt, dim=-1)
            pred_multi_label = torch.unique(feat_pred, dim=-1)

            result = dict(
                img_gt = bs_img_gt[bidx],
                img_pred = bs_img_pred[bidx],
                cls_feat_gt = cls_feat_gt[bidx],
                csl_feat_pred = csl_feat_pred[bidx],
                gt_multi_label = gt_multi_label,
                pred_multi_label = pred_multi_label,
            )

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        result_metrics = dict()

        img_gt = [rs['img_gt'] for rs in results]
        img_pred = [rs['img_pred'] for rs in results]
        feat_gt = torch.stack([rs['cls_feat_gt'] for rs in results]).flatten()
        feat_pred = torch.stack([rs['csl_feat_pred'] for rs in results]).flatten()

        img_result = calculate_metrics(img_gt,img_pred)
        for k,v in img_result.items():
            if k != 'cm':
                result_metrics['img_'+k] = v
        
        token_result = {'precision':[],'recall':[]}
        token_result['acc'] = accuracy_score(feat_gt, feat_pred)
        report = classification_report(feat_gt, feat_pred, output_dict=True)
        for cls, metrics in report.items():
            if isinstance(metrics, dict) and 'avg' not in cls:  # 跳过 'accuracy' 总值
                token_result['precision'].append(metrics['precision'])
                token_result['recall'].append(metrics['recall'])
        token_result['macro_avg'] = report['macro avg']
        for k,v in token_result.items():
            if k != 'cm':
                result_metrics['token_'+k] = v

        gt_multi_label = [rs['gt_multi_label'] for rs in results]
        pred_multi_label = [rs['pred_multi_label'] for rs in results]
        metric_res = self.calculate(
            pred_multi_label,
            gt_multi_label,
            pred_indices=True,
            target_indices=True,
            average=None,
            num_classes=self.num_classes)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics
        
        suffix = '_classwise' if self.thr == 0.5 else f'_thr-{self.thr:.2f}_classwise'
        for k, v in pack_results(*metric_res).items():
            result_metrics[k + suffix] = v.detach().cpu().tolist()
        
        return result_metrics
