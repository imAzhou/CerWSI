import shutil
import os
import sys

target_dir = '/home/zly/codes/cervix_wsi_cls/data_resource/cls_valid/valid'

del_list = []
while True:
    source_img_path = input("请输入地址(按 q 退出)：")
    if source_img_path == 'q':
        for img_path in del_list:
            os.remove(img_path)
        print("清除完毕")
        sys.exit(0)  # 传入退出码，0 表示正常退出，其他值通常表示错误
    else:
        basename = os.path.basename(source_img_path)
        shutil.copy(source_img_path,f'{target_dir}/{basename}')
        del_list.append(source_img_path)
