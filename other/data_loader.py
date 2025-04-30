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

'''Data loader for UCI letter, spam and MNIST datasets.
'''
import os

# Necessary packages
import pandas as pd
import numpy as np
from utils import binary_sampler, calculate_missing_rate
from scipy.special import expit  # sigmoid
import torch


def data_loader(data_name, miss_rate, random_seed=None):
    '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
    # Load data
    global data_x
    if random_seed is not None:
        np.random.seed(random_seed)

        file_name = '../data/months/' + data_name + '.csv'
        data_x = np.loadtxt(file_name, delimiter=",")

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m


def data_loader_mar(
    data_name,
    miss_rate,
    window_size=12,
    stride=1,
    random_seed=None
):
    """
    Loads dataset and introduces MAR missingness using time windows.
    In each window, one random column is kept fully observed; others are partially masked.

    Args:
        data_name (str): dataset name (CSV file without path or suffix)
        miss_rate (float): overall missing rate (0~1)
        window_size (int): time window length
        stride (int): step size for window
        random_seed (int, optional): seed for reproducibility

    Returns:
        data_x (np.ndarray): original data
        miss_data_x (np.ndarray): data with missing values
        data_m (np.ndarray): mask matrix (1=observed, 0=missing)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    file_name = f'../data/months/{data_name}.csv'
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    data_x = np.loadtxt(file_name, delimiter=",")
    n_rows, n_cols = data_x.shape

    data_m = np.ones((n_rows, n_cols), dtype=np.float32)
    total_elements = n_rows * n_cols
    target_missing = int(round(miss_rate * total_elements))
    current_missing = 0

    valid_windows = []

    # Step 1: Collect all valid positions to inject missingness (excluding the protected column in each window)
    for start in range(0, n_rows - window_size + 1, stride):
        end = start + window_size
        protected_col = np.random.randint(n_cols)
        window_indices = []

        for t in range(start, end):
            for c in range(n_cols):
                if c != protected_col:
                    window_indices.append((t, c))

        valid_windows.append(window_indices)

    # Step 2: Flatten all candidate indices and randomly select from them
    all_valid_positions = [pos for win in valid_windows for pos in win]
    np.random.shuffle(all_valid_positions)

    for i in range(len(all_valid_positions)):
        if current_missing >= target_missing:
            break
        r, c = all_valid_positions[i]
        if data_m[r, c] == 1:
            data_m[r, c] = 0
            current_missing += 1

    # Step 3: Apply mask
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m


def data_loader_mnar(data_name, window_size=12, stride=1, offset=0.5, seed=None):
    '''Loads time series data and introduces MNAR missingness (larger values are more likely to be missing),
    based only on offset without explicit missing_rate.

    Args:
        - data_name: CSV filename without path
        - window_size: length of each window for evaluating missingness
        - stride: step size for sliding window
        - offset: controls threshold for missingness (larger offset = fewer values missing)
        - seed: random seed

    Returns:
        - data_x: original data (numpy)
        - miss_data_x: data with MNAR missing values (numpy, with NaNs)
        - data_m: indicator matrix (numpy, 1 = observed, 0 = missing)
    '''
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load data
    file_name = '../data/months/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",")
    data_x_tensor = torch.tensor(data_x, dtype=torch.float32)

    # If 2D, add batch dimension
    if data_x_tensor.ndim == 2:
        data_x_tensor = data_x_tensor.unsqueeze(0)

    X = data_x_tensor.clone()
    n_s, n_l, n_c = X.shape
    ori_mask = (~torch.isnan(X)).float()

    # Initialize full mask (1 = observed)
    missing_mask = torch.ones_like(X)

    for start in range(0, n_l - window_size + 1, stride):
        end = start + window_size
        X_win = X[:, start:end, :]
        mask_win = ori_mask[:, start:end, :]

        # Mean and std for window
        mask_sum = mask_win.sum(1)
        mask_sum[mask_sum == 0] = 1
        mean = (X_win * mask_win).sum(1) / mask_sum
        std = (((X_win - mean.unsqueeze(1)) * mask_win).pow(2).sum(1) / mask_sum).sqrt()
        threshold = mean + offset * std
        threshold = threshold.unsqueeze(1).repeat(1, window_size, 1)

        # Identify where to mask
        cond = (X_win > threshold) & (mask_win == 1)
        indices = cond.nonzero(as_tuple=False)

        for b, t, c in indices:
            missing_mask[b, start + t, c] = 0

    # Apply mask to input
    miss_data_x_tensor = X.clone()
    miss_data_x_tensor[missing_mask == 0] = torch.nan

    # Return (squeezed if originally 2D)
    return (
        X.squeeze(0).numpy(),
        miss_data_x_tensor.squeeze(0).numpy(),
        missing_mask.squeeze(0).numpy()
    )


if __name__ == "__main__":
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        # for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for missing_rate in [0.4]:
            for random_seed in [1, 2, 3]:
                ori_data_x, miss_data_x, data_m = data_loader_mar(data_name, missing_rate, window_size=12, stride=12, random_seed=random_seed)
                miss_radio = calculate_missing_rate(miss_data_x)
                1
                # np.savetxt(f'data/months/{missing_rate}missing_rate/{data_name}'
                #            f'/missingdata_{random_seed}.csv', miss_data_x, delimiter=',')
