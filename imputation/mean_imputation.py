import pandas as pd
import numpy as np
from data_loader import data_loader
from utils import rmse_loss, mae_loss, mape_loss
from sklearn.impute import SimpleImputer


# 均值插补
def mean_imputation(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    columns = data.columns
    for i in columns:
        data[i] = data[i].fillna(data[i].mean())
    data = np.array(data)
    return data


# 创建示例时间序列数据
for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
    for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for random_seed in range(1, 4):
            data_x, miss_data_x, data_m = data_loader(data_name, missing_rate, random_seed)

            # 开始均值插补
            mean = SimpleImputer(strategy="mean")
            imputation_data = mean.fit_transform(miss_data_x)
            # imputation_data = mean_imputation(miss_data_x)

            # 存储评价指标
            rmse = rmse_loss(data_x, imputation_data, data_m)
            mae = mae_loss(data_x, imputation_data, data_m)
            mape = mape_loss(data_x, imputation_data, data_m)
            metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
            metrics = pd.DataFrame(metrics, index=[0])
            metrics.to_csv(f'D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}'
                           f'/mean_{random_seed}.csv', index=False)

            # 存储数据
            np.savetxt(
                f'D:/Desktop/imputation_forecast/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                f'/mean_{random_seed}.csv', imputation_data, delimiter=',')

            # 输出结果
            print(imputation_data)
            print('RMSE Performance: ' + str(np.round(rmse, 4)))
            print('MAE Performance: ' + str(np.round(mae, 4)))
            print('MAPE Performance: ' + str(np.round(mape, 4)))
