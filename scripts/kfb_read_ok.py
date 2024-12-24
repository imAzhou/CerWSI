import glob
import os
from cerwsi.utils import KFBSlide
from tqdm import tqdm

# root_dir = '/medical-data/data/intestine/肠-2024.12.17'
# kfb_list = glob.glob(f'{root_dir}/**/*.kfb')


# error_kfb = []
# for kfb_path in tqdm(kfb_list, ncols=80):
#     try:
#         slide = KFBSlide(kfb_path)
#         width, height = slide.level_dimensions[0]
#         print(f'width:{width}, height:{height}')
#     except:
#         error_kfb.append(f'{kfb_path}\n')

# with open('error_read.txt', 'w') as f:
#     f.writelines(error_kfb)

# with open('error_read.txt', 'r') as f:
#     for line in f.readlines():
#         error_kfb_path = line.strip()
#         os.remove(error_kfb_path)

with open('error_read.txt', 'r') as f:
    for line in f.readlines():
        error_kfb_path = line.strip()
        try:
            slide = KFBSlide(error_kfb_path)
            width, height = slide.level_dimensions[0]
            print(f'width:{width}, height:{height}')
        except:
            print(f' Error: {error_kfb_path}')