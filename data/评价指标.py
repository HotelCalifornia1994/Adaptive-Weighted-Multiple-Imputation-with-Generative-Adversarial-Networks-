import pandas as pd
import numpy as np
from utils import rmse_loss, mae_loss, mape_loss
from other.data_loader import data_loader


for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
    for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for random_seed in [1, 2, 3]:
            for modern_name in ['gain(+l2)', 'gain(+l2+自编码位置编码+多重)', 'gain(+l2+自编码位置编码)', 'gain(+多重)', 'gain(+自编码位置编码)', 'gain(原始)']:
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                # data_x = np.loadtxt(f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\{modern_name}_{random_seed}.csv', delimiter=',')
                data_x = np.loadtxt(f'Ablation\\imputation_data\\{missing_rate}missing_rate\\{data_name}\\{modern_name}_{random_seed}.csv', delimiter=',')
                rmse = rmse_loss(ori_data_x, data_x, data_m)
                mae = mae_loss(ori_data_x, data_x, data_m)
                mape = mape_loss(ori_data_x, data_x, data_m)
                metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
                metrics = pd.DataFrame(metrics, index=[0])
                metrics.to_csv(
                    f"Ablation\\metrics/{missing_rate}missing_rate/{data_name}"
                    f"/{modern_name}_{random_seed}.csv", index=False)
                # 输出结果
                print('RMSE Performance: ' + str(np.round(rmse, 4)))
                print('MAE Performance: ' + str(np.round(mae, 4)))
                print('MAPE Performance: ' + str(np.round(mape, 4)))
