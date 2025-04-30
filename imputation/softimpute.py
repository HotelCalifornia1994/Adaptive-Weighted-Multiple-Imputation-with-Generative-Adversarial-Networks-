from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from data_loader import data_loader
from utils import rmse_loss, mae_loss, mape_loss
import numpy as np
import pandas as pd

# for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
#     for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
for data_name in ['siyue']:
    for missing_rate in [0.7]:
        for random_seed in range(1, 4):
            ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
            imputation_data = SoftImpute().fit_transform(missing_data)
            # 存储数据
            # np.savetxt(f'D:/Desktop/我的插补/data/imputation_data/{missing_rate}missing_rate/{data_name}'
            #            f'/softimpute_{random_seed}.csv', imputation_data, delimiter=',')

            # 存储评价指标
            rmse = rmse_loss(ori_data_x, imputation_data, data_m)
            mae = mae_loss(ori_data_x, imputation_data, data_m)
            mape = mape_loss(ori_data_x, imputation_data, data_m)
            metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
            metrics = pd.DataFrame(metrics, index=[0])
            # metrics.to_csv(f"D:/Desktop/我的插补/data/metrics/{missing_rate}missing_rate/{data_name}"
            #                f"/softimpute_{random_seed}.csv", index=False)

            # 输出最终结果
            print(imputation_data)
            print('RMSE Performance: ' + str(np.round(rmse, 4)))
            print('MAE Performance: ' + str(np.round(mae, 4)))
            print('MAPE Performance: ' + str(np.round(mape, 4)))

