import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import random

# 自定义数据集类
class TokenDataset(Dataset):
    def __init__(self, root_dir, csv_filepath, top_k=256):
        """
        Args:
            root_dir (str): pt file root dir
        """
        self.root_dir = root_dir
        self.top_k = top_k
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
        for row in df_patient.itertuples(index=True):
            if row.Index == self.top_k:
                break
            pt_path = f'{self.root_dir}/pt/{patientId}/{row.pt_filename}'
            token_pt = torch.load(pt_path, map_location=torch.device("cpu"), weights_only=True)
            token_list.append(token_pt)
            start_points.append((row.start_x, row.start_y))

        combined = list(zip(token_list, start_points))
        random.shuffle(combined)
        token_list, start_points = zip(*combined)
        token_list = list(token_list)
        start_points = list(start_points)

        token_tensor = torch.stack(token_list).squeeze(1)   # (k,1,embed_dim) → (k,embed_dim)
        return {
            'inputs': token_tensor, 
            'start_points': start_points,
            'metadata': metadata,
            'label': patient_row.kfb_clsid
        }
