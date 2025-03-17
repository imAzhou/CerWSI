import torch
from mmengine.config import Config
from cerwsi.nets import PatchClsNet

device = torch.device(f'cpu')

d_cfg = Config.fromfile('configs/dataset/l_cerscan_dataset.py')
m_cfg = Config.fromfile('configs/model/resnet.py')
s_cfg = Config.fromfile('configs/strategy.py')

cfg = Config()
for sub_cfg in [d_cfg, s_cfg]:
    cfg.merge_from_dict(sub_cfg.to_dict())

model = PatchClsNet(cfg).to(device)
model.load_backbone(cfg.backbone_ckpt, frozen=cfg.frozen_backbone)