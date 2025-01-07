from torch.utils.data import DataLoader
from cerwsi.datasets import TokenClsDataset
from cerwsi.nets import MultiPatchUNI
import torch

def load_data():
    train_dataset = TokenClsDataset(
        'data_resource/0103', 'train')
    train_loader = DataLoader(train_dataset, 
                            batch_size=16, 
                            shuffle=True, 
                            persistent_workers = True,
                            pin_memory=True,
                            num_workers=16)
    val_dataset = TokenClsDataset(
        'data_resource/0103', 'val')
    val_loader = DataLoader(val_dataset, 
                            batch_size=16, 
                            persistent_workers = True,
                            pin_memory=True,
                            num_workers=16)
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = load_data()
    device = torch.device('cuda:0')
    num_classes = 6
    model = MultiPatchUNI(num_classes, device)

    for dataloader in [train_loader, val_loader]:
        for databatch in dataloader:
            print()