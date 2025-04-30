# Data preprocessing. Tedious, but PyPOTS can help.
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
import tsdb
from pypots.data import load_specific_dataset
data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop(['RecordID', 'Time'], axis=1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_ori = X  # keep X_ori for validation
X = mcar(X, 0.1)  # randomly hold out 10% observed values as ground truth
dataset = {"X": X}  # X for model input
print(X.shape)  # (11988, 48, 37), 11988 samples and each sample has 48 time steps, 37 features

# Model training. This is PyPOTS showtime.
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=10)
# Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
saits.fit(dataset)  # train the model on the dataset
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
saits.save("save_it_here/saits_physionet2012.pypots")  # save the model for future use
saits.load("save_it_here/saits_physionet2012.pypots")  # reload the serialized model file for following imputation or training
