import pandas as pd
import numpy as np
from data_loader import data_loader
data_x, miss_data_x, data_m = data_loader('wind_speed', 0.1, 123)
miss_data_x[np.isnan(miss_data_x)] = 0
U, S, VT = np.linalg.svd(miss_data_x)
np.savetxt('C:/Users/cheems/Desktop/imputation_forecast/data/U.csv', U, delimiter=',')
np.savetxt('C:/Users/cheems/Desktop/imputation_forecast/data/S.csv', S, delimiter=',')
np.savetxt('C:/Users/cheems/Desktop/imputation_forecast/data/VT.csv', VT, delimiter=',')
count_zeros = np.count_nonzero(VT == 0)
print(count_zeros)