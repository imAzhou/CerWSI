import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace


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


class CerMClassifier(nn.Module):
    def __init__(self, num_classes, num_patches, embed_dim):
        '''
        num_classes: positive classes number + 1
        '''
        super(CerMClassifier, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}

        self.instance_loss_fn = nn.CrossEntropyLoss()
        args = {
            'size_type': "big",
            'dropout': True
        }
        args = SimpleNamespace(**args)
        size = self.size_dict[args.size_type]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if args.dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=args.dropout, n_classes=num_classes-1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], 1)

        
    @property
    def device(self):
        return next(self.parameters()).device

    def calc_logits(self, feature_emb: torch.Tensor):
        '''
        Args:
            feature_emb: (bs,cls_token+img_token,C)
        Return:
            img_logits: (bs, 1)
        '''
        img_tokens = feature_emb[:,1:,:]  # (bs, num_tokens, C)
        # A: (bs, num_tokens, pos_cls_num), h: (bs, num_tokens, c=512)
        A, h = self.attention_net(img_tokens)
        A = A.transpose(1, 2)    # A: (bs, pos_cls_num, num_tokens)
        A = F.softmax(A, dim=-1)
        cls_feature = torch.bmm(A, h)    # cls_feature: (bs, pos_cls_num, c=512)
        out = self.classifiers(cls_feature)    # (bs, pos_cls_num, 1)
        return out.squeeze(-1)
    
    def calc_pos_loss(self, pos_logits, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = torch.zeros_like(pos_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            label_list = list(set([tk[-1] -1 for tk in token_labels]))  # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
            binary_matrix[i, label_list] = 1

        loss = loss_fn(pos_logits, binary_matrix)
        return loss
    
    def calc_loss(self,feature_emb, databatch):
        positive_logits = self.calc_logits(feature_emb)
        loss = self.calc_pos_loss(positive_logits, databatch)
        return loss

    def set_pred(self,feature_emb, databatch):
        positive_logits = self.calc_logits(feature_emb) # (bs, num_classes-1)
        databatch['pos_probs'] = torch.sigmoid(positive_logits) # (bs, num_classes-1)
        return databatch
