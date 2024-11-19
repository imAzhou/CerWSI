import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideClsNet(nn.Module):
    def __init__(self, num_classes, input_channels=2048, nhead=8, num_layers=2):
        super(SlideClsNet, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(input_channels, dtype=torch.float32))
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_channels, nhead=nhead, activation='gelu')
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        #seq = torch.cat([self.cls_token.expand(1,x.shape[1],-1), x], dim=0)
        seq = torch.cat([self.cls_token.repeat(1,x.shape[1],1), x], dim=0)
        fea = self.encoder(seq)[0,:] #取输出中cls_token对应的输出特征
        logit = self.fc(fea)
        return logit

    def loss(self, x, gt):
        pass

    def pred(self, x):
        pass
