import torch
from torch import nn
import torch.nn.functional as F
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, BinaryMetric


class BinaryLinear(MetaClassifier):
    def __init__(self, **args):
        input_embed_dim = args.input_embed_dim
        self.num_classes = args.num_classes
        evaluator = build_evaluator([BinaryMetric(thr = args.positive_thr)])
        super(BinaryLinear, self).__init__(evaluator, **args)

        self.cls_linear_head = nn.Linear(input_embed_dim, 1)

    def calc_logits(self, feature_emb: torch.Tensor):
        pred_img_logits = self.cls_linear_head(feature_emb)  # (bs, 1)
        return pred_img_logits
    
    def calc_loss(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb)
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        pn_loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')
        return pn_loss

    def set_pred(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb) # (bs, num_classes-1)
        databatch['img_probs'] = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        return databatch
