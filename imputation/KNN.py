"""Impute missing values with k nearest classifier."""
import sys
import numpy as np
import pandas as pd
from sklearn import neighbors
from data_loader import data_loader
from utils import rmse_loss, mae_loss, mape_loss
from sklearn.impute import KNNImputer


class KNN:
    """Imputer class."""

    def _fit(self, X, column, k=10, is_categorical=False):
        """Fit a knn classifier for missing column.

        - Args:
                X(numpy.ndarray): input data
                column(int): column id to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                clf: trained k nearest neighbour classifier
        """
        clf = None
        if not is_categorical:
            clf = neighbors.KNeighborsRegressor(n_neighbors=k)
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        # use column not null to train the kNN classifier
        missing_idxes = np.where(pd.isnull(X[:, column]))[0]
        if len(missing_idxes) == 0:
            return None
        X_copy = np.delete(X, missing_idxes, 0)
        X_train = np.delete(X_copy, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_train[:, col_id]))[0]
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_train[col_missing_idxes, col_id] = col_mean[col_id]
        y_train = X_copy[:, column]
        # fit classifier
        clf.fit(X_train, y_train)
        return clf

    def _transform(self, X, column, clf, is_categorical):
        """Impute missing values.

        - Args:
                X(numpy.ndarray): input numpy ndarray
                column(int): index of column to be imputed
                clf: pretrained classifier
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X(pandas.dataframe): imputed dataframe
        """
        missing_idxes = np.where(np.isnan(X[:, column]))[0]
        X_test = X[missing_idxes, :]
        X_test = np.delete(X_test, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        # fill missing values in each column with current col_mean
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_test[:, col_id]))[0]
            # if no missing values for current column
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_test[col_missing_idxes, col_id] = col_mean[col_id]
        # predict missing values
        y_test = clf.predict(X_test)
        X[missing_idxes, column] = y_test
        return X

    def knn(self, X, column, k=10, is_categorical=False):
        """Impute missing value with knn.

        - Args:
                X(pandas.dataframe): dataframe
                column(str): column name to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X_imputed(pandas.dataframe): imputed pandas dataframe
        """
        X, column = self._check_X_y(X, column)
        clf = self._fit(X, column, k, is_categorical)
        if clf is None:
            return X
        else:
            X_imputed = self._transform(X, column, clf, is_categorical)
            return X_imputed

    def _check_X_y(self, X, column):
        """Check input, if pandas.dataframe, transform to numpy array.

        - Args:
                X(ndarray/pandas.dataframe): input instances
                column(str/int): column index or column name
        - Returns:
                X(ndarray): input instances
        """
        column_idx = None
        if isinstance(X, pd.core.frame.DataFrame):
            if isinstance(column, str):
                # get index of current column
                column_idx = X.columns.get_loc(column)
            else:
                column_idx = column
            X = X.values
        else:
            column_idx = column
        return X, column_idx


if __name__ == "__main__":
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in range(1, 4):
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                missing_data = pd.DataFrame(missing_data)

                # default estimators are lgbm classifier and regressor
                # list = []
                # colum_list = missing_data.columns
                # for column in colum_list:
                #     knn = KNN()
                #     imputation_colum = knn.knn(X=missing_data, column=column, k=10)
                #     list = imputation_colum
                # imputation_data = np.array(list)
                # 开始调用sklean的KNN插补
                KNN = KNNImputer(n_neighbors=5, weights='uniform')
                imputation_data = KNN.fit_transform(missing_data)
                # 存储数据
                np.savetxt(
                    f'D:/Desktop/imputation_forecast/data/imputation_data/{missing_rate}missing_rate/{data_name}'
                    f'/knn_{random_seed}.csv',
                    imputation_data, delimiter=',')
                # 存储评价指标
                rmse = rmse_loss(ori_data_x, imputation_data, data_m)
                mae = mae_loss(ori_data_x, imputation_data, data_m)
                mape = mape_loss(ori_data_x, imputation_data, data_m)
                metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
                metrics = pd.DataFrame(metrics, index=[0])
                metrics.to_csv(
                    f"D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}"
                    f"/knn_{random_seed}.csv", index=False)
                # 输出结果
                print(imputation_data)
                print('RMSE Performance: ' + str(np.round(rmse, 4)))
                print('MAE Performance: ' + str(np.round(mae, 4)))
                print('MAPE Performance: ' + str(np.round(mape, 4)))
