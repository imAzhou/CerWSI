import torch
from torch import nn
import torch.nn.functional as F
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, MultiPosMetric


class MLCLinear(MetaClassifier):
    def __init__(self, args):
        
        evaluator = build_evaluator([MultiPosMetric(thr = args.positive_thr)])
        super(MLCLinear, self).__init__(evaluator, **args)

        input_embed_dim = args.backbone_output_dim[0]
        self.num_classes = args.num_classes

        self.cls_linear_heads = nn.ModuleList()
        for i in range(self.num_classes-1):  # 只判断 image 中含不含阳性 token
            self.cls_linear_heads.append(nn.Linear(input_embed_dim, 1))

    def calc_logits(self, feature_emb: torch.Tensor):
        pred_img_logits = []
        for i in range(self.num_classes-1):
            pred_img_logits.append(self.cls_linear_heads[i](feature_emb))  # [(bs, 1),]
        pred_img_logits = torch.cat(pred_img_logits, dim=-1)  # (bs, num_cls)
        return pred_img_logits
    
    def calc_loss(self,feature_emb, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        pred_logits = self.calc_logits(feature_emb)
        binary_matrix = databatch['multi_pos_labels'].to(self.device)
        loss = loss_fn(pred_logits, binary_matrix)
        return loss

    def set_pred(self,feature_emb, databatch):
        positive_logits = self.calc_logits(feature_emb) # (bs, num_classes-1)
        databatch['pos_probs'] = torch.sigmoid(positive_logits) # (bs, num_classes-1)
        return databatch
