import torch
import numpy as np
import timm
from mmpretrain import get_model
from cerwsi.nets import MultiPatchUNI

# device = torch.device('cuda:0')
# model = MultiPatchUNI(num_classes = 6).to(device)
# with open('my_model.txt', 'w') as f:
#     f.writelines(model)

# # 获取类别索引
# class_indices = torch.nonzero(class_has_high_prob, as_tuple=False).squeeze(1)

# uni_model = timm.create_model(
#             "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
#         )
# with open('uni_model.txt', 'w') as f:
#     f.writelines(str(uni_model))

# for name, param in uni_model.named_parameters():
#     if 'blocks' not in name:
#         print(name)

# model_name = "vit_large_patch16_224.orig_in21k"
# backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
# print(backbone.default_cfg)

# weights = np.load('checkpoints/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz')
# state_dict = {key: torch.from_numpy(value) for key, value in weights.items()}
# print(backbone.load_state_dict(state_dict, strict=False))

# print(backbone.load_state_dict(torch.load('checkpoints/resnet50_a1_0-14fe96d1.pth', map_location='cpu'), strict=False))
# availableModels = timm.list_models(pretrained=True)
# print(availableModels)
# with open('resnet50_model.txt', 'w') as f:
#     f.writelines(str(backbone))
# print(backbone.num_features)

# vit-large-p14_dinov2-pre_3rdparty
# vit-large-p16_in21k-pre_3rdparty_in1k-384px
model = get_model('vit-large-p14_dinov2-pre_3rdparty', pretrained=False).backbone
ckpt = 'checkpoints/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth'
params_weight = torch.load(ckpt)
new_state_dict = {}
for key,value in params_weight['state_dict'].items():
    new_name = key.replace('backbone.', '')
    new_state_dict[new_name] = value
print(model.load_state_dict(new_state_dict, strict=False))

print(model.embed_dims)
inputs = torch.rand(1, 3, 224, 224)
feats = model(inputs)
print(type(feats))