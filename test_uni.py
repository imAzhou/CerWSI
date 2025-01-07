import timm
import os
import torch
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
from cerwsi.nets import MultiPatchUNI
# login('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')  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "checkpoints/"
# os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)

# uni_model = timm.create_model(
#     "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
# )

# params_weight = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
# uni_model.load_state_dict(params_weight, strict=True)
# uni_model.eval()
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


from PIL import Image
image = Image.open("/x22201018/datasets/MedicalDatasets/CoNIC/train/img_dir/consep_1_0000.png")
print(image.mode)
image = transform(image).unsqueeze(dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)

myModel = MultiPatchUNI(num_classes=6, device=torch.device('cuda:1'))
logits = myModel(image)

# with torch.inference_mode():
#     feature_emb = uni_model.forward_features(image)
#     print()