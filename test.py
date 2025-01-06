import numpy as np
import time
from cerwsi.utils import generate_cut_regions

# 假设网格化切割大小
grid_size = 700
overlap = 50
stride = grid_size - overlap
start_time = time.time()
cut_points = generate_cut_regions((0,0), 98760, 120870, grid_size, stride=stride)
t_delta = time.time() - start_time
print(f'generate_cut_regions cost {t_delta:0.2f}s')

def find_overlapping_bboxes(target_bbox, stride, grid_size):
    x_min, y_min, x_max, y_max = target_bbox

    col_min, col_max = x_min // stride, x_max // stride
    row_min, row_max = y_min // stride, y_max // stride

    matching_bboxes = []  # 存储重叠的 bbox 索引或坐标
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            bbox_x_min = col * stride
            bbox_y_min = row * stride
            bbox_x_max = bbox_x_min + grid_size
            bbox_y_max = bbox_y_min + grid_size
            
            # 检查 target_bbox 是否和当前网格 bbox 存在重叠
            if not (x_max < bbox_x_min or x_min > bbox_x_max or 
                    y_max < bbox_y_min or y_min > bbox_y_max):
                matching_bboxes.append(((bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max), (row, col)))
    
    return matching_bboxes

# 给定 target_bbox 的坐标范围
target_bbox = [25, 599, 40, 400]  # 替换为实际坐标

result = find_overlapping_bboxes(target_bbox, stride, grid_size)
for bbox, (row, col) in result:
    print(f"Target bbox overlaps with grid bbox at row {row}, col {col}, with coords {bbox}")