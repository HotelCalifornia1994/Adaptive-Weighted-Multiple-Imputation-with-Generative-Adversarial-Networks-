# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''

# Necessary packages
import numpy as np
import pandas as pd
import random
# from statsmodels.tsa.stattools import adfuller
import tensorflow as tf


# from statsmodels.tsa.stattools import adfuller


##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization

  Returns:
    - renorm_data: renormalized original data
  '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values

  Returns:
    - rounded_data: rounded imputed data
  '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness

  Returns:
    - rmse: Root Mean Squared Error
  '''

    # ori_data, norm_parameters = normalization(ori_data)
    # imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(float(nominator) / float(denominator))

    return rmse


def mae_loss(ori_data, imputed_data, data_m):
    # ori_data, norm_parameters = normalization(ori_data)
    # imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(abs((1 - data_m) * ori_data - (1 - data_m) * imputed_data))
    denominator = np.sum(1 - data_m)

    mae = float(nominator) / float(denominator)

    return mae


def mape_loss(ori_data, imputed_data, data_m):
    # ori_data, norm_parameters = normalization(ori_data)
    # imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    no, dim = data_m.shape
    nominator = np.sum(abs((1 - data_m) * ori_data - (1 - data_m) * imputed_data) / abs(ori_data))
    denominator = np.sum(1 - data_m)

    mape = float(nominator) / float(denominator)

    return mape


def r2_score(ori_data, imputed_data, data_m):
    ori_mean = np.sum(ori_data * (1 - data_m)) / (np.sum((1 - data_m)))
    ss_res = np.sum((ori_data - imputed_data) ** 2)
    ss_tot = np.sum((ori_data - ori_mean) ** 2)
    return 1 - ss_res / ss_tot


def xavier_init(size):
    """Xavier initialization.

  Args:
    - size: vector size

  Returns:
    - initialized random vector.
  """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    binary_random_matrix[0, :] = 1
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns

  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size):
    """Sample index of the mini-batch.

  Args:
    - total: total number of samples
    - batch_size: batch size

  Returns:
    - batch_idx: batch index
  """
    total_idx = np.random.permutation(total)
    # total_idx = np.arange(total)
    # random_point = np.random.randint(0, total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def series_to_supervised(data, n_in, n_out, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def adf_test(time_series, significance_level=0.05):
    """
    执行ADF检验以确定时间序列的平稳性。

    参数：
    - time_series：要进行ADF检验的时间序列数据（Pandas Series或NumPy数组）。
    - significance_level：显著性水平，用于确定是否拒绝原假设（默认为0.05）。

    返回：
    - result：包含ADF检验的结果的字典。
    """

    # 执行ADF检验
    adf_result = adfuller(time_series, autolag='AIC')

    # 提取ADF检验结果
    result = {
        'Test Statistic': adf_result[0],
        'P-value': adf_result[1],
        'Lags Used': adf_result[2],
        'Number of Observations Used': adf_result[3],
        'Critical Values (1%)': adf_result[4]['1%'],
        'Critical Values (5%)': adf_result[4]['5%'],
        'Critical Values (10%)': adf_result[4]['10%'],
        'Stationary (Reject H0)': adf_result[1] <= significance_level
    }

    return result


def time_delays_matrix(data_m):
    data_m = pd.DataFrame(data_m)
    data_m = data_m.copy()
    colum = data_m.columns
    list = []
    for i in colum:
        data = data_m[i]
        index = data.index
        for j in index:
            if j == 0:
                time = j
                data[j] = 0
            elif j > 0:
                if data[j] == 1:
                    time_x = j
                    data[j] = j - time
                    time = time_x
                elif data[j] == 0:
                    data[j] = j - time
        list.append(data)
    data = np.array(list).T
    # print(data)
    return data


def sequence_batch_index(total, batch_size):
    """Sample index of the mini-batch.

  Args:
    - total: total number of samples
    - batch_size: time step

  Returns:
    - batch_idx: batch index
  """
    total_point = total - batch_size
    start = np.random.randint(0, total_point)
    batch_idx = list(range(start, start + batch_size))
    return batch_idx


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc


def phase_space_reconstruction(time_series, time_step):
    """
    对时间序列数据进行重构
    :param time_series: 原始时间序列数据
    :param lag: 延迟时间
    :return: 重构的相空间矩阵
    """
    global list_temp
    list_temp = list()
    no, dim = time_series.shape
    num_points = no - time_step + 1

    for i in range(num_points):
        data_temp = time_series[i:time_step + i]
        list_temp.append(data_temp)
    result = np.array(list_temp)
    return result


def calculate_missing_rate(data):
    """
    计算缺失数据中的总体缺失率（以 np.nan 为缺失值标记）

    Args:
        data (np.ndarray): 带缺失值的数组

    Returns:
        float: 缺失率（0.0 ~ 1.0 之间）
    """
    total_elements = data.size
    num_missing = np.isnan(data).sum()
    missing_rate = num_missing / total_elements
    return missing_rate


if __name__ == '__main__':
    test = position_encoding_init(20, 20)
    print(test)
