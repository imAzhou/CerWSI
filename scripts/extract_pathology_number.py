from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image
import os
import pandas as pd
from PIL import Image
import glob
import re
from tqdm import tqdm
from natsort import natsorted

def extract_pathoId():
    df_data = pd.read_csv('data_resource/slide_anno/group_csv/WXL_2.csv')
    root_dir = '/medical-data/data/cervix/negative_WSI'
    label_save_dir = f'{root_dir}/pathology_numbers'
    os.makedirs(label_save_dir, exist_ok=True)
    pathology_number = []

    for kfb_path in tqdm(natsorted(glob.glob(f'{root_dir}/**/*.kfb')), ncols=80):
        filename = os.path.basename(kfb_path).split('.')[0]
        parent_dir = os.path.dirname(kfb_path).split('/')[-1]
        slide = KFBSlide(kfb_path)
        # 获取所有关联图像名称
        associated_images = kfbslide_get_associated_image_names(slide._osr)
        if 'label' not in associated_images:
            print(f'No label!')
            continue
        filtered = df_data.loc['/medical-data/data/' + df_data['kfb_path'] == kfb_path].iloc[0]
        patientId = filtered['patientId']

        output_dir =f'{label_save_dir}/{parent_dir}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{filename}.png'
        
        image = kfbslide_read_associated_image(slide._osr, 'label')
        rotated_image = image.transpose(Image.ROTATE_270)  # 右旋90°
        rotated_image.save(output_path, "PNG")

        pathoId = 'pathology_number'
        pattern1 = r"^C\d{9}$"         # 第一种模式: C开头 + 9个数字
        pattern2 = r"^(C\d{9})001$"    # 第二种模式: 第一种模式后面加 "001"
        if re.match(pattern1, filename):
            pathoId = str(filename)  # 直接转字符
        elif match := re.match(pattern2, filename):
            pathoId = str(match.group(1))  # 删除 "001" 并转字符

        pathology_number.append([patientId, kfb_path, pathoId])

    df_patho = pd.DataFrame(pathology_number, columns = ['patientId', 'kfb_path', 'pathology_number'])
    df_patho.to_csv('data_resource/slide_anno/0319/WXL_2_pathology_number.csv',index=False)

def check_duplicates():
    df_data = pd.read_csv('data_resource/slide_anno/0319/WXL_2_pathology_number.csv')
    pathology_counts = df_data["pathology_number"].value_counts()
    # 筛选出出现次数大于 1 的 `pathology_number`
    duplicates = pathology_counts[pathology_counts > 1].index  # 只保留重复的 pathology_number
    df_duplicates = df_data[df_data["pathology_number"].isin(duplicates)]  # 获取所有重复行数据
    os.makedirs(f'statistic_results/duplicates', exist_ok=True)

    # 按 `pathology_number` 分组并遍历
    for pathology_number, group in df_duplicates.groupby("pathology_number"):
        for row in group.itertuples(index=True):
            slide = KFBSlide(row.kfb_path)
            level = len(slide.level_dimensions) - 1
            width,height = slide.level_dimensions[level]
            location, size = (0,0), (width,height)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            read_result.save(f'statistic_results/duplicates/{pathology_number}_{row.Index}.png')

if __name__ == '__main__':
    # extract_pathoId()
    check_duplicates()

'''
setfacl -d -m u::rwx,g::---,o::r-x /zheyi_zhijiang
'''