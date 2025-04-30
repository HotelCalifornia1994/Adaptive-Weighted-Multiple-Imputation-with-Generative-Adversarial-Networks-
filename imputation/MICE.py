# -*- coding: utf-8 -*-
"""
Multiple Imputation by Chained Equations
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
"""

import pandas as pd
import numpy as np
from data_loader import data_loader
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, Lasso
from sklearn.model_selection import train_test_split
from utils import rmse_loss, mae_loss, mape_loss


class MiceImputer(object):

    def __init__(self, seed_values=True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy  # seed_strategy in ['mean','median','most_frequent', 'constant']
        self.seed_values = seed_values  # seed_values = False initializes missing_values using not_null columns
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method='Linear', iter=5, verbose=True):

        # Why use Pandas?
        # http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
        # Pandas < Numpy if X.shape[0] < 50K
        # Pandas > Numpy if X.shape[0] > 500K

        # Data necessary for masking missing-values after imputation
        null_cols = X.columns[X.isna().any()].tolist()
        null_X = X.isna()[null_cols]

        ### Initialize missing_values

        if self.seed_values:

            # Impute all missing values using SimpleImputer 
            if verbose:
                print('Initilization of missing-values using SimpleImputer')
            new_X = pd.DataFrame(self.imp.fit_transform(X))
            new_X.columns = X.columns
            new_X.index = X.index

        else:

            # Initialize a copy based on value of self.copy
            if self.copy:
                new_X = X.copy()
            else:
                new_X = X

            not_null_cols = X.columns[X.notna().any()].tolist()

            if verbose:
                print('Initilization of missing-values using regression on non-null columns')

            for column in null_cols:

                null_rows = null_X[column]
                train_x = new_X.loc[~null_rows, not_null_cols]
                test_x = new_X.loc[null_rows, not_null_cols]
                train_y = new_X.loc[~null_rows, column]

                if X[column].nunique() > 2:
                    m = LinearRegression(n_jobs=-1)
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)

                elif X[column].nunique() == 2:
                    m = LogisticRegression(n_jobs=-1, solver='lbfgs')
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)

        ### Begin iterations of MICE

        model_score = {}

        for i in range(iter):
            if verbose:
                print('Beginning iteration ' + str(i) + ':')

            model_score[i] = []

            for column in null_cols:

                null_rows = null_X[column]
                not_null_y = new_X.loc[~null_rows, column]
                not_null_X = new_X[~null_rows].drop(column, axis=1)

                train_x, val_x, train_y, val_y = train_test_split(not_null_X, not_null_y, test_size=0.3,
                                                                  random_state=42)
                test_x = new_X.drop(column, axis=1)

                if new_X[column].nunique() > 2:
                    if method == 'Linear':
                        m = LinearRegression(n_jobs=-1)
                    elif method == 'Ridge':
                        m = Ridge()
                    elif method == 'Lasso':
                        m = Lasso()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

                elif new_X[column].nunique() == 2:
                    if method == 'Linear':
                        m = LogisticRegression(n_jobs=-1, solver='lbfgs')
                    elif method == 'Ridge':
                        m = RidgeClassifier()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

            if model_score[i] == []:
                model_score[i] = 0
            else:
                model_score[i] = sum(model_score[i]) / len(model_score[i])

        return new_X


if __name__ == '__main__':
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in range(1, 4):
                mice = MiceImputer()
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                missing_data = pd.DataFrame(missing_data)
                imputation_data = mice.fit_transform(missing_data, method='lasso', iter=5, verbose=True)

                # 存储数据
                # np.savetxt(f'D:/Desktop/imputation_forecast/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                #            f'/mice_{random_seed}.csv',
                #            imputation_data, delimiter=',')

                # 存储评价指标
                rmse = rmse_loss(ori_data_x, imputation_data.values, data_m)
                mae = mae_loss(ori_data_x, imputation_data.values, data_m)
                mape = mape_loss(ori_data_x, imputation_data.values, data_m)
                metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
                metrics = pd.DataFrame(metrics, index=[0])
                # metrics.to_csv(f"D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}"
                #                f"/mice_{random_seed}.csv", index=False)

                # 输出最终结果
                print(imputation_data)
                print('RMSE Performance: ' + str(np.round(rmse, 4)))
                print('MAE Performance: ' + str(np.round(mae, 4)))
                print('MAPE Performance: ' + str(np.round(mape, 4)))
