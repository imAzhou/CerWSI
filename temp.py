from mmengine.config import Config
from cerwsi.nets import PatchClsNet
import torch


dataset_config_file = 'configs/dataset/l_cerscanv4_dataset.py'
model_config_file = 'configs/model/wscer_partial.py'
strategy_config_file = 'configs/strategy.py'
d_cfg = Config.fromfile(dataset_config_file)
m_cfg = Config.fromfile(model_config_file)
s_cfg = Config.fromfile(strategy_config_file)

cfg = Config()
for sub_cfg in [d_cfg, m_cfg, s_cfg]:
    cfg.merge_from_dict(sub_cfg.to_dict())

device = torch.device("cuda:7")
model = PatchClsNet(cfg).to(device)
input = torch.rand((6,3,1024,1024)).to(device)
databatch = {'images': input}
output = model(databatch, 'val')