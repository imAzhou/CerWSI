

import json


def load_partial_list():
    with open('data_resource/0416/annofiles/train_partial_pos.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data_resource/0416/annofiles/val_partial_pos.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    train_patchlist,val_patchlist = [],[]
    for kfbinfo in train_data:
        train_patchlist.extend(kfbinfo['patch_list'])
    for kfbinfo in val_data:
        val_patchlist.extend(kfbinfo['patch_list'])
    
    return train_patchlist,val_patchlist

def load_total_list():
    with open('data_resource/0416/annofiles/train_roi_total.json', 'r', encoding='utf-8') as f:
        train_roi_data = json.load(f)
    with open('data_resource/0416/annofiles/val_roi_total.json', 'r', encoding='utf-8') as f:
        val_roi_data = json.load(f)
    with open('data_resource/0416/annofiles/train_slide_total.json', 'r', encoding='utf-8') as f:
        train_slide_data = json.load(f)

    train_patchlist,val_patchlist = [],[]
    for kfbinfo in train_roi_data:
        train_patchlist.extend(kfbinfo['patchlist'])
    for kfbinfo in val_roi_data:
        val_patchlist.extend(kfbinfo['patchlist'])
    for kfbinfo in train_slide_data:
        train_patchlist.extend(kfbinfo['patchlist'])
    
    return train_patchlist,val_patchlist

def load_neg_list():
    with open('data_resource/0416/annofiles/train_negslide_patches.json', 'r', encoding='utf-8') as f:
        train_neg_data = json.load(f)
    with open('data_resource/0416/annofiles/val_negslide_patches.json', 'r', encoding='utf-8') as f:
        val_neg_data = json.load(f)

    train_patchlist,val_patchlist = [],[]
    train_patchlist,val_patchlist = [],[]
    for kfbinfo in train_neg_data:
        train_patchlist.extend(kfbinfo['patch_list'])
    for kfbinfo in val_neg_data:
        val_patchlist.extend(kfbinfo['patch_list'])
    
    return train_patchlist,val_patchlist

def analyze_patchlist(total_patchlist):    
    pn_cnt = [0,0]
    for patchinfo in total_patchlist:
        pn_cnt[patchinfo['diagnose']] += 1
    print(pn_cnt)

def main():
    train_partial, val_partial = load_partial_list()
    train_total,val_total = load_total_list()
    train_neg,val_neg = load_neg_list()

    train_data = [*train_partial, *train_total, *train_neg]
    val_data = [*val_partial, *val_total, *val_neg]

    analyze_patchlist(train_data)
    analyze_patchlist(val_data)

if __name__ == "__main__":
    main()

'''
Neg,Pos
train: [70131, 75382]
val: [17208, 19936]
'''