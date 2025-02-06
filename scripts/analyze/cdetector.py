import json
import cv2
import os

if __name__ == '__main__':
    root_dir = '/x22201018/datasets/CervicalDatasets/ComparisonDetectorDataset'
    
    for mode in ['train', 'test']:
        image_dir = f'{root_dir}/{mode}'
        print(len(os.listdir(image_dir)))
        # w_list,h_list = [],[]
        # for filename in os.listdir(image_dir):
        #     img = cv2.imread(f'{image_dir}/{filename}')
        #     h,w,c = img.shape
        #     w_list.append(w)
        #     h_list.append(h)
        # print(f'mode: {mode}, w: {list(set(w_list))}, h: {list(set(h_list))}')

