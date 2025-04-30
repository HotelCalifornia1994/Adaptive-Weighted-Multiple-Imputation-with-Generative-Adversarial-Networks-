import pandas as pd
import numpy as np
from data_loader import data_loader
from utils import rmse_loss, mae_loss, mape_loss
from tqdm import tqdm


def MVNImputer(df, epsilon=1e-5, maxiter=100):
    data = df.values
    n_obs, n_var = data.shape
    mu_init = np.nanmean(data, axis=0)
    sigma_init = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            vecs = data[:, [i, j]]
            vecs = vecs[~np.any(np.isnan(vecs), axis=1), :].T
            if len(vecs) > 0:
                cov = np.cov(vecs)
                cov = cov[0, 1]
                sigma_init[i, j] = cov
                sigma_init[j, i] = cov
            else:
                sigma_init[i, j] = 1.0
                sigma_init[j, i] = 1.0

    print('Start EM iteration.')
    pre_mu = mu_init
    pre_sigma = sigma_init
    pre_lik = -np.inf
    for n_iter in tqdm(range(maxiter)):
        # E step
        temp_data = np.copy(data)
        for i in range(n_obs):
            if np.any(np.isnan(temp_data[i, :])):
                nans = np.isnan(temp_data[i, :])
                # conditional distribution of multivariate normal
                offset_mu = np.dot(pre_sigma[nans, :][:, ~nans], np.dot(np.linalg.inv(pre_sigma[~nans, :][:, ~nans]),
                                                                        (temp_data[i, ~nans] - pre_mu[~nans])[:,
                                                                        np.newaxis]))
                temp_data[i, nans] = pre_mu[nans] + offset_mu.T
        # M step
        new_mu = np.mean(temp_data, axis=0)
        new_sigma = np.cov(temp_data.T)
        new_lik = -0.5 * n_obs * (n_var * np.log(2 * np.pi) + np.log(np.linalg.det(new_sigma)))
        for i in range(n_obs):
            new_lik -= 0.5 * np.dot((temp_data[i, :] - new_mu),
                                    np.dot(np.linalg.inv(new_sigma), (temp_data[i, :] - new_mu)[:, np.newaxis]))
        print('ITER =', n_iter, '\tLog likelihood =', new_lik)
        if new_lik - pre_lik < epsilon:
            imputed = temp_data
            break
        pre_mu = new_mu
        pre_sigma = new_sigma
        pre_lik = new_lik
    else:
        imputed = temp_data
    print('End EM iteration.\n')
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)


if __name__ == '__main__':
    # ori_data_x, missing_data, data_m = data_loader('yiyue', 0.5, 1)
    # missing_data = pd.DataFrame(missing_data)
    # imputation_data = MVNImputer(missing_data)
    # print(imputation_data)
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in tqdm(range(1, 4)):
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                # 开始调用EM插补
                missing_data = pd.DataFrame(missing_data)
                imputation_data = MVNImputer(missing_data)
                # 存储数据
                # np.savetxt(
                #     f'D:/Desktop/imputation_forecast/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                #     f'/em_{random_seed}.csv', imputation_data, delimiter=',')
                # 存储评价指标
                rmse = rmse_loss(ori_data_x, imputation_data.values, data_m)
                mae = mae_loss(ori_data_x, imputation_data.values, data_m)
                mape = mape_loss(ori_data_x, imputation_data.values, data_m)
                metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
                metrics = pd.DataFrame(metrics, index=[0])
                # metrics.to_csv(
                #     f"D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}"
                #     f"/em_{random_seed}.csv", index=False)
                # 输出结果
                print(imputation_data)
                print('RMSE Performance: ' + str(np.round(rmse, 4)))
                print('MAE Performance: ' + str(np.round(mae, 4)))
                print('MAPE Performance: ' + str(np.round(mape, 4)))
