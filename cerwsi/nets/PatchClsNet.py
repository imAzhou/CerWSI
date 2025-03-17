import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from .backbone import get_backbone
from .neck import get_neck
from .classifier import get_classifier

class PatchClsNet(nn.Module):
    def __init__(self, cfg):
        super(PatchClsNet, self).__init__()

        self.backbone = get_backbone(cfg)
        self.neck = get_neck(cfg)
        self.classifier = get_classifier(cfg)
        
        if cfg.frozen_backbone:
            self.freeze_backbone()
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def freeze_backbone(self):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.load_state_dict(params_weight, strict=True))
    
    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        feature_emb = self.backbone(input_x.to(self.device))
        feature_emb = self.neck(feature_emb)
        loss = self.classifier.calc_loss(feature_emb, databatch)
        optim_wrapper.update_params(loss)
        return loss

    def val_step(self, databatch):
        input_x = databatch['images']
        feature_emb = self.backbone(input_x.to(self.device))
        feature_emb = self.neck(feature_emb)
        databatch = self.classifier.set_pred(feature_emb, databatch)
        return databatch
