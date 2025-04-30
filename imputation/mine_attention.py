import pandas as pd
import numpy as np
import os
from utils import rmse_loss, mae_loss, mape_loss
from utils import normalization, renormalization, rounding, binary_sampler, uniform_sampler, xavier_init, \
    sample_batch_index, sequence_batch_index, position_encoding_init
from data_loader import data_loader
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    # Define mask matrix（定义掩码矩阵）
    data_m = 1 - np.isnan(data_x)
    # 获取数据维度
    no, dim = data_x.shape
    # 定义位置矩阵
    positional = position_encoding_init(no, 64).astype(np.float32)

    # System parameters（传递网络参数）
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Normalization（将数据标准化）
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data)

    # GAIN architecture（生成对抗网络结构）
    # Input placeholders（输入网络占位符，定义网络形状）
    # Data vector（数据向量）
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector（掩码向量）
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector（提示向量）
    H = tf.placeholder(tf.float32, shape=[None, dim])
    # 位置向量
    P = tf.placeholder(tf.float32, shape=[None, 64])

    # Generator variables（生成器变量）
    # 第一层查询向量
    G_Q_W1 = tf.Variable(xavier_init([64, 64]))
    # 第一层键向量
    G_K_W1 = tf.Variable(xavier_init([64, 64]))
    # 第一层值向量
    G_V_W1 = tf.Variable(xavier_init([64, 64]))
    # 调整输入数据维度
    G_W1 = tf.Variable(xavier_init([dim * 2, 64]))
    G_b1 = tf.Variable(tf.zeros(shape=[64]))

    G_W2 = tf.Variable(xavier_init([64, dim]))
    G_b2 = tf.Variable(tf.zeros(shape=dim))

    G_W3 = tf.Variable(xavier_init([dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Discriminator variables（判别器变量）
    # 第一层查询向量
    D_Q_W1 = tf.Variable(xavier_init([64, 64]))
    # 第一层键向量
    D_K_W1 = tf.Variable(xavier_init([64, 64]))
    # 第一层值向量
    D_V_W1 = tf.Variable(xavier_init([64, 64]))

    D_W1 = tf.Variable(xavier_init([dim * 2, 64]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[64]))

    D_W2 = tf.Variable(xavier_init([64, dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[dim]))

    D_W3 = tf.Variable(xavier_init([dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # GAIN functions
    # Generator
    def generator(x, m, p):
        # Concatenate Mask and Data 并添加位置向量
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h1 = tf.add(G_h1, p)
        # G_h1 = tf.keras.layers.LayerNormalization(axis=1)(G_h1)  # 嵌入位置向量后标准化

        # 获取注意力权重矩阵
        Q_1 = tf.nn.relu(tf.matmul(G_h1, G_Q_W1))
        K_1 = tf.nn.relu(tf.matmul(G_h1, G_K_W1))
        V_1 = tf.nn.relu(tf.matmul(G_h1, G_V_W1))
        # 计算注意力权重
        A_1 = tf.matmul(Q_1, tf.transpose(K_1)) / tf.sqrt(tf.cast(64, tf.float32))
        A_1 = tf.nn.softmax(A_1)
        # 计算注意力权重后的值
        O_1 = tf.matmul(A_1, V_1)
        # 合并注意力权重后的值和原始输入
        G_h1 = tf.add(G_h1, O_1)
        # 标准化
        G_h1 = tf.keras.layers.LayerNormalization(axis=1)(G_h1)

        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h, p):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h1 = tf.add(D_h1, p)
        # D_h1 = tf.keras.layers.LayerNormalization(axis=1)(D_h1)  # 嵌入位置向量后标准化

        # 获取注意力权重矩阵
        Q_1 = tf.nn.relu(tf.matmul(D_h1, D_Q_W1))
        K_1 = tf.nn.relu(tf.matmul(D_h1, D_K_W1))
        V_1 = tf.nn.relu(tf.matmul(D_h1, D_V_W1))
        # 计算注意力权重
        A_1 = tf.matmul(Q_1, tf.transpose(K_1)) / tf.sqrt(tf.cast(64, tf.float32))
        A_1 = tf.nn.softmax(A_1)
        # 计算注意力权重后的值
        O_1 = tf.matmul(A_1, V_1)
        # 合并注意力权重后的值和原始输入
        D_h1 = tf.add(D_h1, O_1)
        # 标准化
        D_h1 = tf.keras.layers.LayerNormalization(axis=1)(D_h1)

        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)

        D_prob = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)
        return D_prob

    # GAIN structure
    # Generator
    G_sample = generator(X, M, P)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H, P)

    # 添加L2正则化项
    l2_lambda = gain_parameters['l2_lambda']  # 调整正则化强度
    regularization_G = tf.reduce_sum([tf.nn.l2_loss(var) for var in theta_G])  # 对生成器参数应用L2正则化
    regularization_D = tf.reduce_sum([tf.nn.l2_loss(var) for var in theta_D])

    # GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1 - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss + l2_lambda * regularization_G

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
        batch_idx = sequence_batch_index(no, batch_size)
        # batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        P_mb = positional[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.1, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb, P: P_mb})

        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb, P: P_mb})
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
    P_mb = positional
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    H_mb_temp = binary_sampler(1, no, dim)
    H_mb = M_mb * H_mb_temp

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb, P: P_mb})[0]
    possible = sess.run([D_prob], feed_dict={X: X_mb, M: M_mb, H: H_mb, P: P_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data, possible


def mine(first_imputation_data, ori_data, ori_missing_data, ori_data_m):
    df = first_imputation_data.copy()
    missing_data = ori_missing_data.copy()
    possible_list = list()
    dataset_list = list()
    possible_ma = pd.DataFrame()
    for count in range(5):
        for colum in range(5):
            gain_parameters = {'batch_size': 64,
                               'hint_rate': 0.9,
                               'alpha': 100,
                               'iterations': 10000,
                               'learning_rate': 0.001,
                               'l2_lambda': 0.01}
            df[colum] = missing_data[colum]
            is_missing = np.isnan(df)
            data_m = np.where(is_missing, 0, 1)

            print(f'\033[32m开始第{count + 1}次， 第{colum + 1}列迭代：\033[0m')
            tf.reset_default_graph()
            df, possible = gain(df.values, gain_parameters)
            df = pd.DataFrame(df)
            possible = pd.DataFrame(possible)
            print(df)

            rmse = rmse_loss(ori_data.values, df.values, data_m)
            mae = mae_loss(ori_data.values, df.values, data_m)
            mape = mape_loss(ori_data.values, df.values, data_m)

            print('RMSE Performance: ' + str(np.round(rmse, 4)))
            print('MAE Performance: ' + str(np.round(mae, 4)))
            print('MAPE Performance: ' + str(np.round(mape, 4)))

            # 将生成目标列概率的数据存下等待加权平均
            possible_ma[colum] = possible[colum]

        # 将数据矩阵和概率矩阵保存
        df_temp = df.copy()
        possible_temp = possible_ma.copy()
        dataset_list.append(df_temp)
        possible_list.append(possible_temp)

    # 概率矩阵连加求和
    possible_all = np.zeros_like(possible_list[0])
    for matrix in possible_list:
        possible_all += matrix

    # 创建期望矩阵
    expected_matrix = np.zeros_like(possible_list[0])
    for prob_matrix, data_matrix in zip(possible_list, dataset_list):
        expected_matrix += np.multiply(prob_matrix, data_matrix)

    # 加权平均（最终结果）
    imputed_data_x = expected_matrix / possible_all
    imputed_data_x = pd.DataFrame(imputed_data_x)
    # print('\033[31m最终结果\033[0m')
    # print(imputed_data_x)

    # 评价指标
    rmse = rmse_loss(ori_data.values, imputed_data_x.values, ori_data_m)
    mae = mae_loss(ori_data.values, imputed_data_x.values, ori_data_m)
    mape = mape_loss(ori_data.values, imputed_data_x.values, ori_data_m)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
    metrics = pd.DataFrame(metrics, index=[0])
    # print('RMSE Performance: ' + str(np.round(rmse, 4)))
    # print('MAE Performance: ' + str(np.round(mae, 4)))
    # print('MAPE Performance: ' + str(np.round(mape, 4)))
    return imputed_data_x, metrics


if __name__ == '__main__':
    for data_name in ['yiyue']:
        for missing_rate in [0.7]:
            for random_seed in [1, 2, 3]:
                tf.reset_default_graph()
                first_imputation_data = pd.read_csv(
                    f'D:/Desktop/我的插补/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                    f'/gain_{random_seed}.csv', header=None)
                ori_data, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                missing_data = pd.DataFrame(missing_data)
                ori_data = pd.DataFrame(ori_data)
                imputation_data_final, metrics = mine(first_imputation_data, ori_data, missing_data, data_m)
                print(f'\033[31m这是{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的最终结果\033[0m')
                print(imputation_data_final)
                print('\033[32mRMSE Performance:\033[0m ' + str(np.round(metrics['rmse'].iloc[0], 4)))
                print('\033[32mMAE Performance:\033[0m ' + str(np.round(metrics['mae'].iloc[0], 4)))
                print('\033[32mMAPE Performance:\033[0m ' + str(np.round(metrics['mape'].iloc[0], 4)))

                # 存储评价指标
                # metrics.to_csv(f'D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}/'
                #                f'mine_{random_seed}.csv', index=False)
                # 保存数据
                # imputation_data_final.to_csv(
                #     f'D:/Desktop/imputation_forecast/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                #     f'/mine_{random_seed}.csv', index=False, header=False)
                # print(f'\033[31m{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个样本的插补结果已保存\033[0m')
