from cerwsi.datasets import TokenDataset
from torch.utils.data import DataLoader
from mmengine.config import Config
import torch.multiprocessing as mp

def load_token_dataset(mode, cfg: Config,num_workers=4):
    mp.set_start_method('spawn', force=True)

    config_dict = cfg.dataset_cfg[mode]
    dataset = TokenDataset(config_dict.root_dir, config_dict.csv_filepath)
    dataloader = DataLoader(dataset, 
                            batch_size=config_dict.bs, 
                            shuffle=True, 
                            persistent_workers = True,
                            pin_memory=True,
                            num_workers=num_workers)
    return dataloader