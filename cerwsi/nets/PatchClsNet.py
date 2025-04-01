import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from .get_backbone import get_backbone
from .get_neck import get_neck
from .get_classifier import get_classifier

class PatchClsNet(nn.Module):
    def __init__(self, cfg):
        super(PatchClsNet, self).__init__()

        self.backbone = get_backbone(cfg)
        self.neck = get_neck(cfg)
        self.classifier = get_classifier(cfg)

        frozen_backbone = cfg.backbone_cfg['frozen_backbone']
        use_peft = cfg.backbone_cfg['use_peft']
        self.backbone_nograd = frozen_backbone and use_peft is None

        self.split_group = cfg.split_group
        
    @property
    def device(self):
        return next(self.parameters()).device

    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.load_state_dict(params_weight, strict=True))
    
    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def extract_feature(self, input_x):
        
        if self.backbone_nograd:
            self.backbone.eval()
            with torch.no_grad():
                features = []
                for split_x in torch.chunk(input_x, self.split_group, dim=0):
                    split_x = split_x.to(self.device)
                    features.append(self.backbone(split_x))
                feature_emb = torch.cat(features, dim=0)  # 重新拼接回完整批次
        else:
            feature_emb = self.backbone(input_x)
        return feature_emb

    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        input_x = databatch['images']   # (bs, c, h, w)
        feature_emb = self.extract_feature(input_x)
        feature_emb = self.neck(feature_emb)
        loss,loss_dict = self.classifier.calc_loss(feature_emb, databatch)
        optim_wrapper.update_params(loss)
        return loss,loss_dict

    def val_step(self, databatch):
        input_x = databatch['images']
        feature_emb = self.extract_feature(input_x)
        feature_emb = self.neck(feature_emb)
        databatch = self.classifier.set_pred(feature_emb, databatch)
        return databatch
