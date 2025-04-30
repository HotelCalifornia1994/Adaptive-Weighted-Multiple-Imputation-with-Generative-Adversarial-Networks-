from other.data_loader import data_loader, data_loader_mar, data_loader_mnar
from utils import calculate_missing_rate, rmse_loss, mape_loss, mae_loss, r2_score
from gain_l2_posi_multi import gain, multiple
import os
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from ts_models import SAITS_imputation, BRITS_imputation, USGAN_imputation, proposed_imputation

tf.disable_v2_behavior()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def imputation_experiment_mechanisms(missing_type='mcar', data_name='siyue', missing_rate=0.4, offset=0.3,
                                     random_seed=1):
    if missing_type == 'mcar':
        ori_data, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
    elif missing_type == 'mar':
        ori_data, missing_data, data_m = data_loader_mar(data_name, missing_rate, window_size=12, stride=12,
                                                         random_seed=random_seed)
    elif missing_type == 'mnar':
        ori_data, missing_data, data_m = data_loader_mnar(data_name, window_size=12, stride=12, offset=offset,
                                                          seed=random_seed)
    else:
        raise ValueError("missing_type must be one of 'mcar', 'mar', or 'mnar'.")

    # 核对缺失率
    radio = calculate_missing_rate(missing_data)
    print(f'数据缺失率为:{radio}')

    # experiment for mcar

    # 第一次插补
    gain_first_parameters = {'batch_size': 64,
                             'hint_rate': 0.9,
                             'alpha': 100,
                             'iterations': 10000,
                             'learning_rate': 0.001,
                             'l2_lambda': 0.01}
    first_imputation_data, _ = gain(missing_data, gain_first_parameters)
    tf.reset_default_graph()
    print(
        f'\033[31m{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的第一次插补已完成\033[0m')
    # 多重插补任务开始
    first_imputation_data = pd.DataFrame(first_imputation_data).copy()
    missing_data = pd.DataFrame(missing_data)
    ori_data = pd.DataFrame(ori_data)
    imputation_data_final, metrics = multiple(first_imputation_data, ori_data, missing_data, data_m)
    print(f'\033[31m这是{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的最终结果\033[0m')
    print(imputation_data_final)
    print('\033[32mRMSE Performance:\033[0m ' + str(np.round(metrics['rmse'].iloc[0], 4)))
    print('\033[32mMAE Performance:\033[0m ' + str(np.round(metrics['mae'].iloc[0], 4)))
    print('\033[32mMAPE Performance:\033[0m ' + str(np.round(metrics['mape'].iloc[0], 4)))

    # 存储评价指标
    metrics.to_csv(
        f'../data/Discussion/Different_missing_mechanisms/metrics/{data_name}_{missing_rate}_{missing_type}_gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
        index=False)
    # 保存数据
    imputation_data_final.to_csv(
        f'../data/Discussion/Different_missing_mechanisms/imputation_data/{data_name}_{missing_rate}_{missing_type}_gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
        index=False, header=False)


def aqi_experiment(data_name='Beijing_aqi', missing_rate=0.4, random_seed=1):
    ori_data, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
    # 核对缺失率
    radio = calculate_missing_rate(missing_data)
    print(f'数据缺失率为:{radio}')

    # experiment for mcar
    # 第一次插补
    gain_first_parameters = {'batch_size': 64,
                             'hint_rate': 0.9,
                             'alpha': 100,
                             'iterations': 10000,
                             'learning_rate': 0.001,
                             'l2_lambda': 0.001}
    first_imputation_data, _ = gain(missing_data, gain_first_parameters)
    rmse = rmse_loss(ori_data, first_imputation_data, data_m)
    mae = mae_loss(ori_data, first_imputation_data, data_m)
    mape = mape_loss(ori_data, first_imputation_data, data_m)
    print('预插补RMSE Performance: ' + str(np.round(rmse, 4)))
    print('预插补MAE Performance: ' + str(np.round(mae, 4)))
    print('预插补MAPE Performance: ' + str(np.round(mape, 4)))
    tf.reset_default_graph()
    print(
        f'\033[31m{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的第一次插补已完成\033[0m')

    # 多重插补任务开始
    first_imputation_data = pd.DataFrame(first_imputation_data).copy()

    missing_data = pd.DataFrame(missing_data)
    ori_data = pd.DataFrame(ori_data)
    imputation_data_final, metrics = multiple(first_imputation_data, ori_data, missing_data, data_m)
    print(f'\033[31m这是{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的最终结果\033[0m')
    print(imputation_data_final)
    print('\033[32mRMSE Performance:\033[0m ' + str(np.round(metrics['rmse'].iloc[0], 4)))
    print('\033[32mMAE Performance:\033[0m ' + str(np.round(metrics['mae'].iloc[0], 4)))
    print('\033[32mMAPE Performance:\033[0m ' + str(np.round(metrics['mape'].iloc[0], 4)))

    # 存储评价指标
    metrics.to_csv(
        f'../data/Discussion/aqi/metrics/{data_name}_{missing_rate}_gain(+l2+自编码位置编码+多重)_aqi_{random_seed}.csv',
        index=False)
    # 保存数据
    imputation_data_final.to_csv(
        f'../data/Discussion/aqi/imputation_data/{data_name}_{missing_rate}_gain(+l2+自编码位置编码+多重)_aqi_{random_seed}.csv',
        index=False, header=False)


if __name__ == '__main__':
    # aqi_experiment(data_name='Beijing_aqi', missing_rate=0.1, random_seed=1)
    # aqi_experiment(data_name='Beijing_aqi', missing_rate=0.2, random_seed=1)
    # aqi_experiment(data_name='Beijing_aqi', missing_rate=0.3, random_seed=1)
    aqi_experiment(data_name='Beijing_aqi', missing_rate=0.4, random_seed=1)

