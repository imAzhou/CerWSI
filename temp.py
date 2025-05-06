import json
import pandas as pd
from tqdm import tqdm
import os
import shutil

with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI_filter.json', 'r', encoding='utf-8') as f:
    roi_data = json.load(f)
df_data_0409 = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
df_data_0422 = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
df_data = pd.concat([df_data_0409, df_data_0422])

drop_pids = []
for item in roi_data:
    imageName = item['imageName']
    pid = '_'.join(imageName.split('_')[:3])
    drop_pids.append(pid)
for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
    drop_pids.append(row.patientId)

for mode in ['train', 'val']:
    df_origin = pd.read_csv(f'data_resource/0429_2/annofiles/{mode}.csv')
    new_data = []
    for row in tqdm(df_origin.itertuples(index=False), total=len(df_origin), ncols=80):
        if row.patientId not in drop_pids:
            new_data.append(row)
    df_new_data = pd.DataFrame(new_data, columns = df_origin.columns)
    df_new_data.to_csv(f'data_resource/0429_2/annofiles/partial_{mode}.csv', index=False)
