import torch
from torch import nn
import torch.nn.functional as F
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, MyMultiClsMetric


class MultiClsLinear(MetaClassifier):
    def __init__(self, args):
        input_embed_dim = args.input_embed_dim
        self.num_classes = args.num_classes
        evaluator = build_evaluator([MyMultiClsMetric(args.logger_name,num_classes = args.num_classes)])
        super(MultiClsLinear, self).__init__(evaluator, **args)

        self.cls_linear_head = nn.Linear(input_embed_dim, self.num_classes)

    def calc_logits(self, feature_emb: torch.Tensor):
        pred_img_logits = self.cls_linear_head(feature_emb)  # (bs, 1)
        return pred_img_logits
    
    def calc_loss(self,feature_emb, databatch):
        loss_fn = nn.CrossEntropyLoss()
        img_pn_logit = self.calc_logits(feature_emb)
        img_gt = databatch['image_labels'].to(self.device)
        multicls_loss = loss_fn(img_pn_logit, img_gt)
        loss_dict = {
            'multicls_loss': multicls_loss.item(),
        }
        return multicls_loss,loss_dict

    def set_pred(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb) # (bs, num_classes-1)
        img_pn_logit = F.softmax(img_pn_logit, dim=-1)
        databatch['pre_cls'] = torch.max(img_pn_logit, dim=-1)[1]
        return databatch
