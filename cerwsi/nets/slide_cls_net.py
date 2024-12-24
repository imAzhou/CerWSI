import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper


class SlideClsNet(nn.Module):
    def __init__(self, num_classes, device, input_channels=2048, nhead=8, num_layers=2):
        super(SlideClsNet, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(input_channels, dtype=torch.float32))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_channels, nhead=nhead, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_channels, num_classes)

        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        '''
        x.shape: (bs, token_nums, token_dim)
        '''
        seq = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
        fea = self.encoder(seq)[:,0]
        logit = self.fc(fea)
        return logit

    def train_step(self, inputs, gt_label, optim_wrapper:OptimWrapper):

        logits = self.forward(inputs)
        loss = self.loss_fn(logits, gt_label)
        optim_wrapper.update_params(loss)

        return loss

    def val_step(self, inputs):

        cls_score = self.forward(inputs)
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
        return pred_scores,pred_labels
    