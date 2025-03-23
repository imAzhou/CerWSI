import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .cls_dataset import ClsDataset

def load_data(cfg):
    def custom_collate(batch):
        # 拆分 batch 中的图像和标签
        images = [item[0] for item in batch]  # 所有 image_tensor，假设 shape 一致
        image_labels = [item[1] for item in batch]
        multi_pos_labels = [item[2] for item in batch]
        token_labels = [item[3] for item in batch]
        image_paths = [item[4] for item in batch]

        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)
        multilabels_tensor = torch.stack(multi_pos_labels, dim=0)

        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'multi_pos_labels': multilabels_tensor,
            'token_labels': token_labels,
            'image_paths': image_paths
        }

    train_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_dataset = ClsDataset(cfg.data_root, cfg.train_annojson, train_transform, cfg.num_classes)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                            pin_memory=True,
                            batch_size=cfg.train_bs, 
                            sampler = train_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_dataset = ClsDataset(cfg.data_root, cfg.val_annojson, val_transform, cfg.num_classes)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            pin_memory=True,
                            batch_size=cfg.val_bs, 
                            sampler = val_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    
    return train_loader, val_loader

