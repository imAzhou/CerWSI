import re
import pandas as pd

log_file_1 = 'log/1127_train_rcp_c6/test_wsi.log'
log_file_2 = 'log/1127_train_rcp_c6_2/test_wsi.log'

# 用于存储提取的数据
parsed_data = []

# 正则表达式匹配
pattern = re.compile(
    r"positive:(\d+)\snegative:(\d+).*?pred/gt:(\d+)/(\d+)"
)

# 逐行读取并解析日志
# kfb_clsid,p_path_num,n_patch_num
for log_file in [log_file_1, log_file_2]:
    with open(log_file, 'r') as file:
        for line in file:
            # 去掉空行或无效行
            line = line.strip()
            if not line:
                continue

            # 匹配数据
            match = pattern.search(line)
            if match:
                positive = int(match.group(1))
                negative = int(match.group(2))
                pred = int(match.group(3))
                gt = int(match.group(4))
                parsed_data.append([gt, positive, negative])

df_pred_pn = pd.DataFrame(parsed_data, columns = ['kfb_clsid','p_path_num','n_patch_num'])
df_pred_pn.to_csv('log/1127_train_rcp_c6/pred_pn.csv', index=False)
