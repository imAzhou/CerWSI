from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler,default_collate
from torch.utils.data.distributed import DistributedSampler
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpretrain.datasets import MultiLabelDataset


def load_multilabel_dataset(*, d_config: Config, seed: int, distributed: bool = False):

    init_default_scope('mmpretrain')

    train_dataset_config = d_config['train_dataloader']['dataset']
    train_dataset = MultiLabelDataset(**train_dataset_config)

    val_dataset_config = d_config['val_dataloader']['dataset']
    val_dataset = MultiLabelDataset(**val_dataset_config)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = DefaultSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DefaultSampler(val_dataset, shuffle=True, seed=seed)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = d_config['train_dataloader']['batch_size'],
        num_workers = d_config['train_dataloader']['num_workers'],
        sampler = train_sampler,
        collate_fn = default_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = d_config['val_dataloader']['batch_size'],
        num_workers = d_config['val_dataloader']['num_workers'],
        sampler = val_sampler,
        collate_fn = default_collate
    )

    return train_dataloader,val_dataloader
