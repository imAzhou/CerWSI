from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler,default_collate
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.datasets import MultiLabelDataset


def load_multilabel_dataset(*, d_config: Config, seed: int):

    init_default_scope('mmpretrain')

    train_dataset_config = d_config['train_dataloader']['dataset']
    train_dataset = MultiLabelDataset(**train_dataset_config)

    val_dataset_config = d_config['val_dataloader']['dataset']
    val_dataset = MultiLabelDataset(**val_dataset_config)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = d_config['train_dataloader']['batch_size'],
        num_workers = d_config['train_dataloader']['num_workers'],
        sampler = DefaultSampler(train_dataset, shuffle=True, seed=seed),
        collate_fn = default_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = d_config['val_dataloader']['batch_size'],
        num_workers = d_config['val_dataloader']['num_workers'],
        sampler = DefaultSampler(val_dataset, shuffle=True, seed=seed),
        collate_fn = default_collate
    )

    return train_dataloader,val_dataloader
