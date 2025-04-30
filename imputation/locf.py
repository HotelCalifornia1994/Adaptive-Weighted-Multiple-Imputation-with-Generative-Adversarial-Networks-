import numpy as np
from data_loader import data_loader
from utils import series_to_supervised, phase_space_reconstruction
from utils import rmse_loss, mae_loss, mape_loss
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS, Transformer, LOCF, Mean
from pypots.utils.metrics import calc_mae

# for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
#     for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
for data_name in ['yiyue']:
    for missing_rate in [0.6]:
        for random_seed in [1, 2, 3]:
            ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
            train_data = phase_space_reconstruction(missing_data, missing_data.shape[0])
            dataset = {"X": train_data}
            locf = LOCF()
            imputation_data = locf.predict(dataset)['imputation'].reshape(-1, 5)
            # 存储数据
            np.savetxt(
                f'D:/Desktop/我的插补/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                f'/locf_{random_seed}.csv', imputation_data, delimiter=',')
            # 存储评价指标
            rmse = rmse_loss(ori_data_x, imputation_data, data_m)
            mae = mae_loss(ori_data_x, imputation_data, data_m)
            mape = mape_loss(ori_data_x, imputation_data, data_m)
            metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
            metrics = pd.DataFrame(metrics, index=[0])
            metrics.to_csv(
                f"D:/Desktop/我的插补/data/metrics/{missing_rate}missing_rate/{data_name}"
                f"/locf_{random_seed}.csv", index=False)
            # 输出结果
            print(imputation_data)
            print('RMSE Performance: ' + str(np.round(rmse, 4)))
            print('MAE Performance: ' + str(np.round(mae, 4)))
            print('MAPE Performance: ' + str(np.round(mape, 4)))
