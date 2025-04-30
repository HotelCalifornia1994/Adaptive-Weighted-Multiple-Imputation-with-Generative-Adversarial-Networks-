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

"""GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, sequence_batch_index, position_encoding_init
import matplotlib.pyplot as plt
import os
import argparse
from data_loader import data_loader
from utils import rmse_loss, mape_loss, mae_loss
from utils import time_delays_matrix
# import tensorflow as tf
# IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def gain(data_x, gain_parameters):
    tf.set_random_seed(123)
    np.random.seed(123)
    '''Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape
    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data)
    # GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])


    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[dim]))

    D_W2 = tf.Variable(xavier_init([dim, dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[dim]))

    D_W3 = tf.Variable(xavier_init([dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))

    # D_W4 = tf.Variable(xavier_init([dim, dim]))
    # D_b4 = tf.Variable(tf.zeros(shape=[dim]))

    theta_D = [D_W1, D_W2, D_W3,
               D_b1, D_b2, D_b3,]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[dim]))

    G_W2 = tf.Variable(xavier_init([dim, dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[dim]))

    G_W3 = tf.Variable(xavier_init([dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    # G_W4 = tf.Variable(xavier_init([dim, dim]))
    # G_b4 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3,
               G_b1, G_b2, G_b3,]

    # GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)

        G_h2 = tf.matmul(G_h1, G_W2) + G_b2
        G_h2 = tf.nn.relu(G_h2)
        # G_h2 = tf.nn.dropout(G_h2, 0.9)

        G_h3 = tf.matmul(G_h2, G_W3) + G_b3
        # G_h3 = tf.nn.relu(G_h3)

        # G_h4 = tf.matmul(G_h3, G_W4) + G_b4
        G_prob = tf.nn.sigmoid(G_h3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)

        D_h2 = tf.matmul(D_h1, D_W2) + D_b2
        D_h2 = tf.nn.relu(D_h2)
        # D_h2 = tf.nn.dropout(D_h2, 0.9)

        D_h3 = tf.matmul(D_h2, D_W3) + D_b3
        # D_h3 = tf.nn.relu(D_h3)

        # D_h4 = tf.matmul(D_h3, D_W4) + D_b4
        D_prob = tf.nn.sigmoid(D_h3)
        return D_prob

    # GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    # GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.math.log(D_prob + 1e-8) + (1 - M) * tf.math.log(1 - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.math.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    # GAIN solver
    learning_rate = gain_parameters['learning_rate']
    D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

    # Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    D_loss_curr_list = []
    G_loss_curr_list = []
    MSE_loss_curr_list = []
    # Start Iterations
    for _ in tqdm(range(iterations)):
        # Sample batch
        # batch_idx = sequence_batch_index(no, batch_size)
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.1, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})

        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
        D_loss_curr_list.append(D_loss_curr)
        G_loss_curr_list.append(G_loss_curr)
        MSE_loss_curr_list.append(MSE_loss_curr)

    # plt.plot(D_loss_curr_list, label='D_loss_curr')
    # plt.plot(G_loss_curr_list, label='G_loss_curr')
    # plt.plot(MSE_loss_curr_list, label='MSE_loss_curr')
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # Return imputed data
    Z_mb = uniform_sampler(0, 0.1, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    H_mb_temp = binary_sampler(1, no, dim)
    H_mb = M_mb * H_mb_temp

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb, H: H_mb})[0]
    possible = sess.run([D_prob], feed_dict={X: X_mb, M: M_mb, H: H_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data, possible



def main(args):
    """
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

  Returns:
    - imputed_data_x: imputed data
    """
    # tf.set_random_seed(123)
    # np.random.seed(123)
    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations,
                       'learning_rate': args.learning_rate}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, args.random_seed)

    # Impute missing data
    imputed_data_x, possible = gain(miss_data_x, gain_parameters)

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    mae = mae_loss(ori_data_x, imputed_data_x, data_m)
    mape = mape_loss(ori_data_x, imputed_data_x, data_m)
    # 输出评价结果
    # print('RMSE Performance: ' + str(np.round(rmse, 4)))
    # print('MAE Performance: ' + str(np.round(mae, 4)))
    # print('MAPE Performance: ' + str(np.round(mape, 4)))
    # 创建评价指标字典
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
    metrics = pd.DataFrame(metrics, index=[0])
    return imputed_data_x, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['yiyue', 'siyue', 'qiyue', 'shiyue'],
        default='yiyue',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        default=0.6,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=10000,
        type=int)
    parser.add_argument(
        '--learning_rate',
        help='learning_rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--random_seed',
        help='random_seed',
        choices=[1, 2, 3],
        default=3,
        type=int)
    args = parser.parse_args()

    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # for data_name in ['siyue']:
    #     for missing_rate in [0.6]:
            for random_seed in [1, 2, 3]:
                tf.reset_default_graph()
                args.data_name = data_name
                args.miss_rate = missing_rate
                args.random_seed = random_seed
                imputed_data, metrics = main(args)
                # 输出插补数据
                print(f'\033[31m这是{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的结果\033[0m')
                print(imputed_data)
                # 输出评价结果
                print('\033[32mRMSE Performance:\033[0m ' + str(np.round(metrics['rmse'].iloc[0], 4)))
                print('\033[32mMAE Performance:\033[0m ' + str(np.round(metrics['mae'].iloc[0], 4)))
                print('\033[32mMAPE Performance:\033[0m ' + str(np.round(metrics['mape'].iloc[0], 4)))
                # 存储评价指标
                metrics.to_csv(f'D:/Desktop/我的插补/data/Ablation/metrics/{missing_rate}missing_rate/{data_name}'
                               f'/gain(原始)_{random_seed}.csv', index=False)
                # 存储数据
                np.savetxt(f'D:/Desktop/我的插补/data/Ablation/imputation_data/{missing_rate}missing_rate/{data_name}'
                           f'/gain(原始)_{random_seed}.csv', imputed_data, delimiter=',')
