import torch
import torch.nn as nn

input = torch.randn((6,64,64,768))
fl_input = input.flatten(start_dim=1,end_dim=2)
print(fl_input.shape)
