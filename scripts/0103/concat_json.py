import json

if __name__ == '__main__':
    
    for mode in ['train','val']:
        datalist = []
        for clstype in ['pos','neg']:
            with open(f'data_resource/0103/annofiles/{mode}_{clstype}_patches.json', 'r') as f:
                pdata = json.load(f)

            for patchInfo in pdata['patch_list']:   # for in each slide patch list
                datalist.append({
                    'filename': patchInfo['filename'],
                    'diagnose': int(patchInfo['diagnose']),
                    'gtmap_14': patchInfo['gtmap_14']
                })
    
        with open(f'data_resource/0103/annofiles/{mode}_patches.json', 'w') as f:
            json.dump(datalist, f)
            