import pandas as pd

# 创建一个示例的DataFrame，其中包含一个时间戳列
df = pd.read_csv('date_data/date.csv', names=['date'])

# 将时间戳列转换为时间数据类型
df['date'] = pd.to_datetime(df['date'])

# 提取年、月、日、小时和分钟
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df = df.drop(['date'], axis=1)

# 打印结果
print(df)

df.to_csv('date_data/date_splitting.csv', index=False)
