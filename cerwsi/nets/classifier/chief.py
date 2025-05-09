import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,BinaryMetric


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CHIEF(MetaClassifier):
    def __init__(self, args):
        num_classes = 1 # 只能做阴阳二分类
        evaluator = build_evaluator([BinaryMetric(args.logger_name, thr = args.positive_thr)])
        super(CHIEF, self).__init__(evaluator, **args)

        self.size_dict = {
            'xs': [384, 256, 256], 
            "small": [768, 512, 256], 
            "big": [1024, 512, 384], 
            'large': [2048, 1024, 512]
        }
        args = SimpleNamespace(**args)
        size = self.size_dict[args.size_type]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if args.dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=args.dropout, n_classes=num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], 1)


    def calc_logits(self, img_tokens: torch.Tensor):
        '''
        Args:
            img_tokens: (bs,img_token,C)
        Return:
            img_logits: (bs, 1)
        '''
        # A: (bs, num_tokens, pos_cls_num), h: (bs, num_tokens, c=512)
        A, h = self.attention_net(img_tokens)
        A = A.transpose(1, 2)    # A: (bs, 1, num_tokens)
        A = F.softmax(A, dim=-1)
        cls_feature = torch.bmm(A, h)    # cls_feature: (bs, 1, c=512)
        out = self.classifiers(cls_feature)    # (bs, 1, 1)
        return out.squeeze(-1)
    
    def calc_loss(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb)
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        pn_loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')
        loss_dict = {
            'pn_loss': pn_loss.item(),
        }
        return pn_loss,loss_dict

    def set_pred(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb) # (bs, num_classes-1)
        databatch['img_probs'] = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        return databatch
