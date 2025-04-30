import numpy as np
from PyEMD import EMD
from data_loader import data_loader
import random

data_x, miss_data_x, data_m = data_loader('wind_speed', 0.4, 123)

# 创建 EMD 对象
WTG03 = EMD()
WTG04 = EMD()
WTG05 = EMD()
WTG06 = EMD()
WTG07 = EMD()
WTG08 = EMD()
WTG09 = EMD()
WTG10 = EMD()
WTG11 = EMD()


# 执行 EMD 分解
np.random.seed(123)
WTG3_IMFs = WTG03(data_x[:, 0])
WTG4_IMFs = WTG04(data_x[:, 1])
WTG5_IMFs = WTG05(data_x[:, 2])
WTG6_IMFs = WTG06(data_x[:, 3])
WTG7_IMFs = WTG07(data_x[:, 4])
WTG8_IMFs = WTG08(data_x[:, 5])
WTG9_IMFs = WTG09(data_x[:, 6])
WTG10_IMFs = WTG10(data_x[:, 7])
variable_dict = {
    "WTG3_IMFs": WTG3_IMFs,
    "WTG4_IMFs": WTG4_IMFs,
    "WTG5_IMFs": WTG5_IMFs,
    "WTG6_IMFs": WTG6_IMFs,
    "WTG7_IMFs": WTG7_IMFs,
    "WTG8_IMFs": WTG8_IMFs,
    "WTG9_IMFs": WTG9_IMFs,
    "WTG10_IMFs": WTG10_IMFs,
    }

# IMFs 现在包含分解后的本征模态函数
for j in range(3, 11):
    data_name = f'WTG{j}_IMFs'
    data = variable_dict[data_name]
    for i, imf in enumerate(data):
        imf_data = imf
        filename = "data/ori_emd/" + f"{data_name}" + "_" + f"{i+1}.csv"
        print(data_name + f"{i + 1}: {imf}")
        np.savetxt(filename, imf_data, delimiter=',')


