import torch
import timm
from mmengine.config import Config
from cerwsi.nets import MultiPatchUNI
from cerwsi.utils import get_parameter_number,get_train_strategy

device = torch.device('cuda:0')
model = MultiPatchUNI(num_classes = 4).to(device)
model.load_backbone_with_LoRA('checkpoints/pytorch_model.bin')
# with open('my_model.txt', 'w') as f:
#     f.writelines(str(model))
# s_cfg = Config.fromfile('configs/train_strategy.py')
# optimizer,lr_scheduler = get_train_strategy(model, s_cfg)
# for epoch in range(s_cfg.max_epochs):
#     current_lr = optimizer.param_groups[0]["lr"]
#     print(f'Epoch: {epoch}, LR: {current_lr:.6f}')
#     lr_scheduler.step()


# current_lr = optimizer.param_groups[0]["lr"]
# print(f'current_lr:{current_lr}')

parameter_cnt = get_parameter_number(model)
print(f'total params: {parameter_cnt}')
for name,parameters in model.named_parameters():
    if not parameters.requires_grad:
        print(name)


# uni_model = timm.create_model(
#             "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
#         )
# with open('uni_model.txt', 'w') as f:
#     f.writelines(str(uni_model))

# for name, param in uni_model.named_parameters():
#     if 'blocks' not in name:
#         print(name)
