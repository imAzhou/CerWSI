import glob
from cerwsi.utils import KFBSlide
from tqdm import tqdm

root_dir = '/disk/medical_datasets/cervix/JFSW_1109'
kfb_list = glob.glob(f'{root_dir}/**/*.kfb')


error_kfb = []
for kfb_path in tqdm(kfb_list, ncols=80):
    try:
        slide = KFBSlide(kfb_path)
        width, height = slide.level_dimensions[0]
        print(f'width:{width}, height:{height}')
    except:
        error_kfb.append(f'{kfb_path}\n')

with open('JFSW_1109_error_read.txt', 'w') as f:
    f.writelines(error_kfb)