import torch, torchvision
import torch.nn as nn
from cerwsi.nets.CHIEF import CHIEF


model = CHIEF(size_arg="small", dropout=True, n_classes=2)

model.eval()
