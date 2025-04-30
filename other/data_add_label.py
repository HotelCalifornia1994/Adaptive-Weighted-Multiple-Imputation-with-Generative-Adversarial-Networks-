import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/imputation_data/0.1missing_rate/gain.csv', header=None)
df = pd.read_csv('data/imputation_data/0.2missing_rate/gain.csv', header=None)
df = pd.read_csv('data/imputation_data/0.3missing_rate/gain.csv', header=None)
df = pd.read_csv('data/imputation_data/0.4missing_rate/gain.csv', header=None)
df = pd.read_csv('data/imputation_data/0.5missing_rate/gain.csv', header=None)
df = pd.read_csv('mean.csv')
# 添加标签
df.columns = ['WTG3', 'WTG4', 'WTG5', 'WTG6', 'WTG7', 'WTG8', 'WTG9', 'WTG10', 'WTG11']
# 读取date.csv文件
df_date = pd.read_csv('date_data/date.csv', header=None)
# 将date.csv的数据插入到data.csv的第一列
df.insert(0, 'date', df_date)
# 将带有标签的DataFrame写回CSV文件
# df.to_csv('data/imputation_data/0.5missing_rate/gain.csv', index=False)
df.to_csv('mean.csv', index=False)