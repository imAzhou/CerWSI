import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace
from .Query2Label.transformer import build_transformer
from .Query2Label.position_encoding import build_position_encoding
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, MultiPosMetric

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
    
class Query2Label(MetaClassifier):
    def __init__(self, **args):
        
        input_embed_dim = args.input_embed_dim
        num_classes = args.num_classes - 1 # 只检测阳性类别
        evaluator = build_evaluator([MultiPosMetric(thr = args.positive_thr)])
        super(Query2Label, self).__init__(evaluator, **args)

        local_args = {
            'feat_size': 14,
            'hidden_dim': 2048,
            'dim_feedforward': 8192,
            'enc_layers': 1,
            'dec_layers': 2,
            'nheads': 4,
            'position_embedding': 'sine',
            'dropout': 0.1,
            'pre_norm': False,
            'keep_other_self_attn_dec': False,
            'keep_first_self_attn_dec': False,
        }
        local_args = SimpleNamespace(**local_args)
        self.transformer = build_transformer(local_args)
        self.position_embedding = build_position_encoding(local_args)
        hidden_dim = self.transformer.d_model
        self.input_proj = nn.Conv2d(input_embed_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def calc_logits(self, img_tokens: torch.Tensor):
        bs,num_tokens,C = img_tokens.shape
        feat_size = int(math.sqrt(num_tokens))
        inputx = img_tokens.reshape(bs, C, feat_size, feat_size)
        pos_emd = self.position_embedding(inputx).to(inputx.dtype)
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(inputx), query_input, pos_emd)[0] # B,K,d
        out = self.fc(hs[-1])  # (bs, num_classes-1)
        return out
    
    def calc_loss(self,feature_emb, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        positive_logits = self.calc_logits(feature_emb)
        binary_matrix = databatch['multi_pos_labels'].to(self.device)
        loss = loss_fn(positive_logits, binary_matrix)
        return loss

    def set_pred(self,feature_emb, databatch):
        positive_logits = self.calc_logits(feature_emb) # (bs, num_classes-1)
        databatch['pos_probs'] = torch.sigmoid(positive_logits) # (bs, num_classes-1)
        return databatch
