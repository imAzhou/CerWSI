from cerwsi.datasets import TokenClsDataset
from torch.utils.data import DataLoader

def load_data():
    train_dataset = TokenClsDataset(
        '/x22201018/datasets/CervicalDatasets/data_resource', 'train')
    train_loader = DataLoader(train_dataset, 
                            batch_size=16, 
                            shuffle=True, 
                            persistent_workers = True,
                            pin_memory=True,
                            num_workers=16)
    val_dataset = TokenClsDataset(
        '/x22201018/datasets/CervicalDatasets/data_resource', 'val')
    val_loader = DataLoader(val_dataset, 
                            batch_size=16, 
                            persistent_workers = True,
                            pin_memory=True,
                            num_workers=16)
    return train_loader, val_loader

if __name__ == '__main__':
    pass