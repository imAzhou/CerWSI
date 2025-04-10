import json
from tqdm import tqdm
import os
import glob
import shutil

total_path = glob.glob(f'/c22073/zly/datasets/CervicalDatasets/LCerScanv3/0403jfsw/images/Pos/Pos_temp/*.png')
for path in tqdm(total_path):
    basename = os.path.basename(path)
    shutil.move(
        path,
        f'/c22073/zly/datasets/CervicalDatasets/LCerScanv3/0403jfsw/images/Pos/{basename}'
    ) 