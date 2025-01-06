import json
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = '/x22201018/datasets/CervicalDatasets/data_resource/annofile'
    with open(f'{root_dir}/1231_train_ann.json', 'r') as f:
        train_ann = json.load(f)
    with open(f'{root_dir}/1231_val_ann.json', 'r') as f:
        val_ann = json.load(f)

    for mode,datalist in zip(['train','val'],[train_ann['data_list'], val_ann['data_list']]):
        lines = []
        for dataitem in tqdm(datalist, ncols=80):
            clsid = 0 if 0 in dataitem['gt_label'] else 1
            lines.append(f'{dataitem["img_path"]} {clsid} \n')
        with open(f'{root_dir}/1231_{mode}_c2.txt', 'w') as f:
            f.writelines(lines)
