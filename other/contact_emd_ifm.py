import numpy as np
variable_dict = {}
for i in range(1, 11):
    for j in range(3, 11):
        file_name = 'C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG{j}_IMFs_{i}' + '.csv'
        IMF = np.loadtxt(file_name, delimiter=",").reshape(-1, 1)
        variable_dict[file_name] = IMF
# horizontal_stacks = []
for n in range(1, 11):
    WTG3_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG3_IMFs_{n}' + '.csv']
    WTG4_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG4_IMFs_{n}' + '.csv']
    WTG5_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG5_IMFs_{n}' + '.csv']
    WTG6_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG6_IMFs_{n}' + '.csv']
    WTG7_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG7_IMFs_{n}' + '.csv']
    WTG8_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG8_IMFs_{n}' + '.csv']
    WTG9_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG9_IMFs_{n}' + '.csv']
    WTG10_IMF = variable_dict['C:/Users/cheems/Desktop/imputation_forecast/data/ori_emd/' + f'WTG10_IMFs_{n}' + '.csv']
    horizontal_stack = np.hstack((WTG3_IMF, WTG4_IMF, WTG5_IMF, WTG6_IMF, WTG7_IMF, WTG8_IMF, WTG9_IMF, WTG10_IMF))
    # horizontal_stacks.append(horizontal_stack)
    np.savetxt(f'C:/Users/cheems/Desktop/imputation_forecast/data/IMF{n}.csv', horizontal_stack, delimiter=',')