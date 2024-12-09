import os
import shutil
from tqdm import tqdm

def count_files(folder):
    """统计文件夹中所有文件的数量（递归）"""
    total_files = 0
    for root, _, files in os.walk(folder):
        total_files += len(files)
    return total_files

def copy_folder_with_progress(source_folder, target_folder):
    """带进度条的文件夹复制"""
    # 统计总文件数
    total_files = count_files(source_folder)
    
    # 初始化进度条
    with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
        def copy_recursive(src, dst):
            """递归复制文件夹内容"""
            if not os.path.exists(dst):
                os.makedirs(dst)
            for item in os.listdir(src):
                source_path = os.path.join(src, item)
                target_path = os.path.join(dst, item)
                if os.path.isdir(source_path):
                    # 如果是文件夹，递归调用
                    copy_recursive(source_path, target_path)
                else:
                    # 如果是文件，复制并更新进度条
                    shutil.copy2(source_path, target_path)
                    pbar.update(1)

        # 开始递归复制
        copy_recursive(source_folder, target_folder)

# 示例使用
source_folder = "/disk/medical_datasets/cervix"  # 替换为实际的源文件夹路径
target_folder = "/medical-data/data/cervix"  # 替换为实际的目标文件夹路径

copy_folder_with_progress(source_folder, target_folder)