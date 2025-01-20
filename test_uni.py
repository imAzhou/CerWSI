import torch
import timm
from cerwsi.nets import MultiPatchUNI

device = torch.device('cuda:0')
model = MultiPatchUNI(num_classes = 6).to(device)
with open('my_model.txt', 'w') as f:
    f.writelines(model)

# 获取类别索引
class_indices = torch.nonzero(class_has_high_prob, as_tuple=False).squeeze(1)

uni_model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
with open('uni_model.txt', 'w') as f:
    f.writelines(str(uni_model))

# for name, param in uni_model.named_parameters():
#     if 'blocks' not in name:
#         print(name)
