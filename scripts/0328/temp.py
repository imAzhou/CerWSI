import os
import numpy as np
import shutil
import glob


output_base = 'data_resource/0328/'  # 输出根目录

# 获取并排序文件，确保文件名一一对应
img_files = sorted(glob.glob(f'data_resource/0328/RoI_label/**/img/*.png'))
ann_files = sorted(glob.glob(f'data_resource/0328/RoI_label/**/ann/*.json'))

# 确保两个文件夹文件数量一致
assert len(img_files) == len(ann_files), "img 和 ann 文件数量不匹配！"

# 按 4 组分割
img_splits = np.array_split(img_files, 4)
ann_splits = np.array_split(ann_files, 4)

# 遍历分组并移动文件
for i in range(4):
    group_name = f'group{i+1}'
    group_path = os.path.join(output_base, group_name)

    # 创建 group 目录及其子目录
    img_output_dir = os.path.join(group_path, 'img')
    ann_output_dir = os.path.join(group_path, 'ann')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(ann_output_dir, exist_ok=True)

    # 移动文件到各自的子目录
    for img_file, ann_file in zip(img_splits[i], ann_splits[i]):
        purename = os.path.basename(str(img_file)).split('.')[0]
        shutil.copy(str(img_file), os.path.join(img_output_dir, f'{purename}.png'))
        shutil.copy(str(ann_file), os.path.join(ann_output_dir, f'{purename}.json'))

    print(f"Group {i+1}: {len(img_splits[i])} items moved.")

print("File splitting and moving completed!")