import json
from tqdm import tqdm
import os
import shutil

classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']

pn_cnt = [0,0]
# for mode in ['train', 'val']:
for mode in ['val']:
    with open(f'data_resource/0328/annofiles/{mode}_patches_v0328.json', 'r') as f:
        patch_list = json.load(f)
    mode_pn_cnt = [0,0]
    all_neg_imgs = os.listdir('data_resource/0328/images/Neg')
    for patchinfo in tqdm(patch_list, ncols=80):
        pn_cnt[patchinfo['diagnose']] += 1
        mode_pn_cnt[patchinfo['diagnose']] += 1
        # if random.random() > 0.01:
        #     continue
        if patchinfo['diagnose'] == 0 and patchinfo['filename'] not in all_neg_imgs:
            shutil.move(
                f'data_resource/0328/images/Pos/{patchinfo["filename"]}',
                f'data_resource/0328/images/Neg/{patchinfo["filename"]}'
            )
        # new_clsname,new_bbox = [],[]
        # for sub_class,bbox in zip(patchinfo['clsnames'], patchinfo['bboxes']):

        #         new_clsname.append(sub_class)
        #         new_bbox.append(bbox)
        # patchinfo['clsnames'] = new_clsname
        # patchinfo['bboxes'] = new_bbox
        # if len(new_bbox) == 0:
        #     patchinfo['diagnose'] = 0

#     with open(f'data_resource/0328/annofiles/{mode}_patches_v0328.json', 'w') as f:
#         json.dump(patch_list,f)
#     print(f'mode {mode}: {mode_pn_cnt}')
# print(pn_cnt)
