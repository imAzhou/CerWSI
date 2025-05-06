import json

RECORD_CLASS = {
    'NILM':'NILM',
    'GEC':'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

with open('data_resource/0429_2/annofiles/partial_train_pos_512.json', 'r', encoding='utf-8') as f:
    partial_train_data = json.load(f)
with open('data_resource/0429_2/annofiles/partial_val_pos_512.json', 'r', encoding='utf-8') as f:
    partial_val_data = json.load(f)

with open(f'data_resource/0429_2/annofiles/0409_zheyi_slide_train_512.json', 'r', encoding='utf-8') as f:
    slide_train_0409 = json.load(f)
with open(f'data_resource/0429_2/annofiles/0422_zheyi_slide_train_512.json', 'r', encoding='utf-8') as f:
    slide_train_0422 = json.load(f)
with open(f'data_resource/0429_2/annofiles/0422_zheyi_slide_val_512.json', 'r', encoding='utf-8') as f:
    slide_val_0422 = json.load(f)
with open(f'data_resource/0429_2/annofiles/roi_train_512.json', 'r', encoding='utf-8') as f:
    roi_train = json.load(f)
with open(f'data_resource/0429_2/annofiles/roi_val_512.json', 'r', encoding='utf-8') as f:
    roi_val = json.load(f)

train_data = []
for patientItem in partial_train_data:
    train_data.extend(patientItem['patch_list'])
for patientItem in slide_train_0409:
    train_data.extend(patientItem['patchlist'])
for patientItem in slide_train_0422:
    train_data.extend(patientItem['patchlist'])
for patientItem in roi_train:
    train_data.extend(patientItem['patchlist'])
   
new_train_data = []
for patchinfo in train_data:
    patchinfo['clsnames'] = [RECORD_CLASS[clsname] for clsname in patchinfo['clsnames']]
    new_train_data.append(patchinfo)

val_data = []
for patientItem in partial_val_data:
    val_data.extend(patientItem['patch_list'])
for patientItem in slide_val_0422:
    val_data.extend(patientItem['patchlist'])
for patientItem in roi_val:
    val_data.extend(patientItem['patchlist'])

new_val_data = []
for patchinfo in val_data:
    patchinfo['clsnames'] = [RECORD_CLASS[clsname] for clsname in patchinfo['clsnames']]
    new_val_data.append(patchinfo)

with open('data_resource/0429_2/annofiles/train.json', 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)
with open('data_resource/0429_2/annofiles/val.json', 'w', encoding='utf-8') as f:
    json.dump(new_val_data, f, ensure_ascii=False, indent=4)
