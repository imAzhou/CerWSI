import pandas as pd
from tqdm import tqdm
import os
import shutil

data_rootdir = 'predict_results'
desc_dir = 'data_resource/0319/patientImgs'
for patientId in tqdm(os.listdir(data_rootdir), ncols=80):
    os.makedirs(f'{desc_dir}/{patientId}', exist_ok=True)
    for filename in os.listdir(f'{data_rootdir}/{patientId}/valid'):
        shutil.move(
            f'{data_rootdir}/{patientId}/valid/{filename}',
            f'{desc_dir}/{patientId}/{filename}'
        )