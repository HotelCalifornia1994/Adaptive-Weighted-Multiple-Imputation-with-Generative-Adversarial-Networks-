import numpy as np
from other.data_loader import data_loader
from utils import series_to_supervised, phase_space_reconstruction
from utils import rmse_loss, mae_loss, mape_loss
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS, BRITS, USGAN
from pypots.utils.metrics import calc_mae
from utils import normalization, renormalization
from utils import rmse_loss, mae_loss, mape_loss
import torch
from gain_l2_posi_multi import gain, multiple
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def SAITS_imputation(data_name, missing_rate, random_seed):
    np.random.seed(123)
    torch.manual_seed(123)
    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
    missing_data_norm, norm_parameters = normalization(missing_data)
    train_data = missing_data_norm.reshape(-1, 10, 5)
    dataset = {"X": train_data}
    saits = SAITS(n_steps=10, n_features=5, n_layers=2, d_model=160, n_heads=5, d_k=32, d_v=32, d_ffn=32, dropout=0,
                  epochs=100, batch_size=64)
    saits.fit(dataset)  # train the model on the dataset
    imputation_norm = saits.impute(dataset).reshape(-1, 5)
    imputation_data = renormalization(imputation_norm, norm_parameters)
    rmse = rmse_loss(ori_data_x, imputation_data, data_m)
    mae = mae_loss(ori_data_x, imputation_data, data_m)
    mape = mape_loss(ori_data_x, imputation_data, data_m)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
    metrics = pd.DataFrame(metrics, index=[0])
    print('RMSE Performance: ' + str(np.round(rmse, 4)))
    print('MAE Performance: ' + str(np.round(mae, 4)))
    print('MAPE Performance: ' + str(np.round(mape, 4)))
    # 存储评价指标
    metrics.to_csv(f'../data/Discussion/ts_compare/metrics/{missing_rate}missing_rate/{data_name}/'
                   f'SAITS_{random_seed}.csv', index=False)
    # 保存数据
    np.savetxt(
        f'../data/Discussion/ts_compare/imputation_data/{missing_rate}missing_rate/{data_name}'
        f'/SAITS_{random_seed}.csv', imputation_data, delimiter=',')


def BRITS_imputation(data_name, missing_rate, random_seed):
    np.random.seed(123)
    torch.manual_seed(123)
    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
    missing_data_norm, norm_parameters = normalization(missing_data)
    train_data = missing_data_norm.reshape(-1, 10, 5)
    dataset = {"X": train_data}
    brits = BRITS(n_steps=10, n_features=5, rnn_hidden_size=32, batch_size=64, epochs=100)
    brits.fit(dataset)  # train the model on the dataset
    imputation_norm = brits.impute(dataset).reshape(-1, 5)
    imputation_data = renormalization(imputation_norm, norm_parameters)
    rmse = rmse_loss(ori_data_x, imputation_data, data_m)
    mae = mae_loss(ori_data_x, imputation_data, data_m)
    mape = mape_loss(ori_data_x, imputation_data, data_m)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
    metrics = pd.DataFrame(metrics, index=[0])
    print('RMSE Performance: ' + str(np.round(rmse, 4)))
    print('MAE Performance: ' + str(np.round(mae, 4)))
    print('MAPE Performance: ' + str(np.round(mape, 4)))
    # 存储评价指标
    metrics.to_csv(f'../data/Discussion/ts_compare/metrics/{missing_rate}missing_rate/{data_name}/'
                   f'BRITS_{random_seed}.csv', index=False)
    # 保存数据
    np.savetxt(
        f'../data/Discussion/ts_compare/imputation_data/{missing_rate}missing_rate/{data_name}'
        f'/BRITS_{random_seed}.csv', imputation_data, delimiter=',')


def USGAN_imputation(data_name, missing_rate, random_seed):
    np.random.seed(123)
    torch.manual_seed(123)
    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
    missing_data_norm, norm_parameters = normalization(missing_data)
    train_data = missing_data_norm.reshape(-1, 10, 5)
    dataset = {"X": train_data}
    usgan = USGAN(n_steps=10, n_features=5, lambda_mse=10, rnn_hidden_size=32, hint_rate=0.9, dropout=0, batch_size=64,
                  epochs=100)
    usgan.fit(dataset)  # train the model on the dataset
    imputation_norm = usgan.impute(dataset).reshape(-1, 5)
    imputation_data = renormalization(imputation_norm, norm_parameters)
    rmse = rmse_loss(ori_data_x, imputation_data, data_m)
    mae = mae_loss(ori_data_x, imputation_data, data_m)
    mape = mape_loss(ori_data_x, imputation_data, data_m)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
    metrics = pd.DataFrame(metrics, index=[0])
    print('RMSE Performance: ' + str(np.round(rmse, 4)))
    print('MAE Performance: ' + str(np.round(mae, 4)))
    print('MAPE Performance: ' + str(np.round(mape, 4)))
    # 存储评价指标
    metrics.to_csv(f'../data/Discussion/ts_compare/metrics/{missing_rate}missing_rate/{data_name}/'
                   f'USGAN_{random_seed}.csv', index=False)
    # 保存数据
    np.savetxt(
        f'../data/Discussion/ts_compare/imputation_data/{missing_rate}missing_rate/{data_name}'
        f'/USGAN_{random_seed}.csv', imputation_data, delimiter=',')


def proposed_imputation(data_name, missing_rate, random_seed):
    np.random.seed(123)
    torch.manual_seed(123)
    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
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
    ori_data = pd.DataFrame(ori_data_x)
    imputation_data_final, metrics = multiple(first_imputation_data, ori_data, missing_data, data_m)
    print(f'\033[31m这是{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的最终结果\033[0m')
    print(imputation_data_final)
    print('\033[32mRMSE Performance:\033[0m ' + str(np.round(metrics['rmse'].iloc[0], 4)))
    print('\033[32mMAE Performance:\033[0m ' + str(np.round(metrics['mae'].iloc[0], 4)))
    print('\033[32mMAPE Performance:\033[0m ' + str(np.round(metrics['mape'].iloc[0], 4)))
    # 存储评价指标
    metrics.to_csv(f'../data/Discussion/ts_compare/metrics/{missing_rate}missing_rate/{data_name}/'
                   f'proposed_{random_seed}.csv', index=False)
    # 保存数据
    np.savetxt(
        f'../data/Discussion/ts_compare/imputation_data/{missing_rate}missing_rate/{data_name}'
        f'/proposed_{random_seed}.csv', imputation_data_final, delimiter=',')
