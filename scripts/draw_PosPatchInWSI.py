from cerwsi.utils import KFBSlide,draw_OD
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

kfb_path = '/nfs-medical/vipa-medical/zheyi/zly/KFBs/till_0318/LSIL/C2024044492025-01-20_16_27_16.kfb'
slide = KFBSlide(kfb_path)
width,height = slide.level_dimensions[-1]
downsample_ratio = slide.level_downsamples[-1]

location, level, size = (0, 0), 5, (width, height)
read_result = Image.fromarray(slide.read_region(location, level, size))
# read_result.save('C2024044492025.png')

with open('log/debug/heat_value/ZY_ONLINE_1_75.json', 'r') as f:
    predInfo = json.load(f)

clsname_arr = ['ASCUS', 'LSIL', 'ASCH', 'HSIL', 'AGC']
colors = plt.cm.tab10(np.linspace(0, 1, len(clsname_arr)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(clsname_arr, colors)}

inside_items = []
for patchinfo in predInfo:
    start_point_x = patchinfo['point'][0] / downsample_ratio
    start_point_y = patchinfo['point'][1] / downsample_ratio
    w = h = 700 / downsample_ratio
    pos_confi = patchinfo['confi_list'][2:]
    clsid = pos_confi.index(max(pos_confi))
    if pos_confi[clsid] > 0.5:
        clsname = clsname_arr[clsid]
        inside_items.append(
            dict(sub_class=clsname,region=dict(
                x=start_point_x,y=start_point_y,width=w,height=h))
        )

square_coords = [0,0,width,height]
draw_OD(read_result,'C2024044492025.png',square_coords,inside_items,category_colors)