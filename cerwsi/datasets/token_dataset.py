import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

# 自定义数据集类
class TokenDataset(Dataset):
    def __init__(self, root_dir, csv_filepath):
        """
        Args:
            root_dir (str): pt file root dir
        """
        self.root_dir = root_dir
        self.patientId_list = os.listdir(f'{root_dir}/pt')
        self.df_data = pd.read_csv(csv_filepath)

    def __len__(self):
        return len(self.patientId_list)

    def __getitem__(self, idx):
        patientId = self.patientId_list[idx]
        df_patient = pd.read_csv(f'{self.root_dir}/pt_record/{patientId}.csv')
        filtered = self.df_data.loc[self.df_data['patientId'] == patientId]
        if filtered.empty:
            print(f"No matching {patientId} found.")
        patient_row = filtered.iloc[0]
        metadata = dict(
            patientId = patientId,
            slide_clsname = patient_row.kfb_clsname,
            kfb_source = patient_row.kfb_source)

        token_list = []
        start_points = []
        for row in df_patient.itertuples(index=False):
            pt_path = f'{self.root_dir}/pt/{patientId}/{row.pt_filename}'
            token_list.append(torch.load(pt_path))
            start_points.append((row.start_x, row.start_y))

        token_tensor = torch.stack(token_list).squeeze(1)   # (k,1,embed_dim) → (k,embed_dim)
        return {
            'inputs': token_tensor, 
            'start_points': start_points,
            'metadata': metadata,
            'label': patient_row.kfb_clsid
        }
