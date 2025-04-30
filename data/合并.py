import os
import pandas as pd
import copy


def combine_data():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'D:/Desktop/我的插补/data/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    f.append(file.split('.')[0])
                    print(file)
                print('-------------------------')

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            # 将每组元素拼接为一行
            space = pd.DataFrame('', index=range(1), columns=range(1))
            space.columns = [' ']
            data_update_x = data_list_update[0].copy()
            data_update = pd.concat([data_update_x[0].copy(), space], axis=1)
            for j in range(1, len(data_update_x)):
                data_update = pd.concat([data_update, data_update_x[j].copy(), space], axis=1)
            # data_update = pd.concat(data_update, axis=1)
            for i in range(1, len(data_list_update)):
                data_update_temp_x = data_list_update[i].copy()
                data_update_temp = pd.concat([data_update_temp_x[0].copy(), space], axis=1)
                for j in range(1, len(data_update_temp_x)):
                    data_update_temp = pd.concat([data_update_temp, data_update_temp_x[j].copy(), space], axis=1)
                # data_update_temp = pd.concat(data_update_temp, axis=1)
                data_update = pd.concat([data_update, data_update_temp])
            data_update.index = mode_list_update
            # if data_name == 'yiyue':
            #     data_update.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/冬季数据（一月）/{data_name}_{missing_rate}all.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'siyue':
            #     data_update.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/春季数据（四月）/{data_name}_{missing_rate}all.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'qiyue':
            #     data_update.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/夏季数据（七月）/{data_name}_{missing_rate}all.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'shiyue':
            #     data_update.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/秋季数据（十月）/{data_name}_{missing_rate}all.csv',
            #         encoding='utf-8-sig')
        # data_update.to_csv(f'D:/Desktop/imputation_forecast/data/metrics/{missing_rate}missing_rate/{data_name}_all.csv')


def combine_data_ablation():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'D:/Desktop/我的插补/data/Ablation/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    f.append(file.split('.')[0])
                    print(file)
                print('-------------------------')

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            # 将每组元素拼接为一行
            space = pd.DataFrame('', index=range(1), columns=range(1))
            space.columns = [' ']
            data_update_x = data_list_update[0].copy()
            data_update = pd.concat([data_update_x[0].copy(), space], axis=1)
            for j in range(1, len(data_update_x)):
                data_update = pd.concat([data_update, data_update_x[j].copy(), space], axis=1)
            # data_update = pd.concat(data_update, axis=1)
            for i in range(1, len(data_list_update)):
                data_update_temp_x = data_list_update[i].copy()
                data_update_temp = pd.concat([data_update_temp_x[0].copy(), space], axis=1)
                for j in range(1, len(data_update_temp_x)):
                    data_update_temp = pd.concat([data_update_temp, data_update_temp_x[j].copy(), space], axis=1)
                # data_update_temp = pd.concat(data_update_temp, axis=1)
                data_update = pd.concat([data_update, data_update_temp])
            data_update.index = mode_list_update
            if data_name == 'yiyue':
                data_update.to_csv(
                    f'D:/Desktop/我的插补/data/Ablation/metrics/冬季数据（一月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'siyue':
                data_update.to_csv(
                    f'D:/Desktop/我的插补/data/Ablation/metrics/春季数据（四月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'qiyue':
                data_update.to_csv(
                    f'D:/Desktop/我的插补/data/Ablation/metrics/夏季数据（七月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'shiyue':
                data_update.to_csv(
                    f'D:/Desktop/我的插补/data/Ablation/metrics/秋季数据（十月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')


def combine_data_regularization():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'../data/regularization_factor/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    filename, ext = os.path.splitext(file)
                    f.append(filename)
                    print(file)
                print('-------------------------')

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            # 将每组元素拼接为一行
            space = pd.DataFrame('', index=range(1), columns=range(1))
            space.columns = [' ']
            data_update_x = data_list_update[0].copy()
            data_update = pd.concat([data_update_x[0].copy(), space], axis=1)
            for j in range(1, len(data_update_x)):
                data_update = pd.concat([data_update, data_update_x[j].copy(), space], axis=1)
            # data_update = pd.concat(data_update, axis=1)
            for i in range(1, len(data_list_update)):
                data_update_temp_x = data_list_update[i].copy()
                data_update_temp = pd.concat([data_update_temp_x[0].copy(), space], axis=1)
                for j in range(1, len(data_update_temp_x)):
                    data_update_temp = pd.concat([data_update_temp, data_update_temp_x[j].copy(), space], axis=1)
                # data_update_temp = pd.concat(data_update_temp, axis=1)
                data_update = pd.concat([data_update, data_update_temp])
            data_update.index = mode_list_update
            if data_name == 'yiyue':
                data_update.to_csv(
                    f'../data/regularization_factor/metrics/冬季数据（一月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'siyue':
                data_update.to_csv(
                    f'../data/regularization_factor/metrics/春季数据（四月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'qiyue':
                data_update.to_csv(
                    f'../data/regularization_factor/metrics/夏季数据（七月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')
            elif data_name == 'shiyue':
                data_update.to_csv(
                    f'../data/regularization_factor/metrics/秋季数据（十月）/{data_name}_{missing_rate}all.csv',
                    encoding='utf-8-sig')


def metrics_average():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        data_month = []
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'D:/Desktop/我的插补/data/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            data_average = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    f.append(file.split('.')[0])

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            for i in range(0, len(data_list_update)):
                data_temp = data_list_update[i]
                df1 = data_temp[0]
                df2 = data_temp[1]
                df3 = data_temp[2]
                df_mean = (df1 + df2 + df3) / 3
                data_average.append(df_mean)
            data_mean = data_average[0].copy()
            for j in range(1, len(data_average)):
                data_mean = pd.concat([data_mean, data_average[j]], axis=0)
            data_month.append(data_mean)
            data_mean.index = mode_list_update
            # if data_name == 'yiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/冬季数据（一月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'siyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/春季数据（四月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'qiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/夏季数据（七月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'shiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/metrics/秋季数据（十月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
        data_month_mean = data_month[0].copy()
        for j in range(1, len(data_month)):
            data_month_mean = pd.concat([data_month_mean, data_month[j]], axis=1)
        if data_name == 'yiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/metrics/冬季数据（一月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'siyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/metrics/春季数据（四月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'qiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/metrics/夏季数据（七月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'shiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/metrics/秋季数据（十月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')


def metrics_average_ablation():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        data_month = []
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'D:/Desktop/我的插补/data/Ablation/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            data_average = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    f.append(file.split('.')[0])

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            for i in range(0, len(data_list_update)):
                data_temp = data_list_update[i]
                df1 = data_temp[0]
                df2 = data_temp[1]
                df3 = data_temp[2]
                df_mean = (df1 + df2 + df3) / 3
                data_average.append(df_mean)
            data_mean = data_average[0].copy()
            for j in range(1, len(data_average)):
                data_mean = pd.concat([data_mean, data_average[j]], axis=0)
            data_month.append(data_mean)
            data_mean.index = mode_list_update
            # if data_name == 'yiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/Ablation/metrics/冬季数据（一月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'siyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/Ablation/metrics/春季数据（四月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'qiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/Ablation/metrics/夏季数据（七月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
            # elif data_name == 'shiyue':
            #     data_mean.to_csv(
            #         f'D:/Desktop/我的插补/data/Ablation/metrics/秋季数据（十月）平均/{data_name}_{missing_rate}average.csv',
            #         encoding='utf-8-sig')
        data_month_mean = data_month[0].copy()
        for j in range(1, len(data_month)):
            data_month_mean = pd.concat([data_month_mean, data_month[j]], axis=1)
        if data_name == 'yiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/Ablation/metrics/冬季数据（一月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'siyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/Ablation/metrics/春季数据（四月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'qiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/Ablation/metrics/夏季数据（七月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'shiyue':
            data_month_mean.to_csv(
                f'D:/Desktop/我的插补/data/Ablation/metrics/秋季数据（十月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')


def metrics_average_regularization():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        data_month = []
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f'../data/regularization_factor/metrics/{missing_rate}missing_rate/{data_name}'
            data = []
            f = []
            numble_list = []
            mode_list = []
            data_average = []
            files = os.listdir(folder_path)
            for file in files:  # 遍历文件夹
                if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才获取
                    path = folder_path + "\\" + file  # 文件地址
                    data.append(pd.read_csv(path))
                    filename, ext = os.path.splitext(file)
                    f.append(filename)

            # 拆分数据名称
            for name in f:
                mode_name = name.split('_')[0]
                number_name = name.split('_')[1]
                numble_list.append(number_name)
                mode_list.append(mode_name)

            # 模型名称去重
            mode_list_update = []
            for item in mode_list:
                # 如果元素不在新列表中，则添加到新列表中
                if item not in mode_list_update:
                    mode_list_update.append(item)

            # 将元素以三个为一组，组成新的列表
            data_list_update = [data[i:i + 3] for i in range(0, len(data), 3)]
            for i in range(0, len(data_list_update)):
                data_temp = data_list_update[i]
                df1 = data_temp[0]
                df2 = data_temp[1]
                df3 = data_temp[2]
                df_mean = (df1 + df2 + df3) / 3
                # 添加列名前缀：当前 missing_rate
                df_mean.columns = [f"{missing_rate}_{col}" for col in df_mean.columns]
                data_average.append(df_mean)
            data_mean = data_average[0].copy()
            for j in range(1, len(data_average)):
                data_mean = pd.concat([data_mean, data_average[j]], axis=0)
            data_month.append(data_mean)
            data_mean.index = mode_list_update
            if data_name == 'yiyue':
                data_mean.to_csv(
                    f'../data/regularization_factor/metrics/冬季数据（一月）平均/{data_name}_{missing_rate}average.csv',
                    encoding='utf-8-sig')
            elif data_name == 'siyue':
                data_mean.to_csv(
                    f'../data/regularization_factor/metrics/春季数据（四月）平均/{data_name}_{missing_rate}average.csv',
                    encoding='utf-8-sig')
            elif data_name == 'qiyue':
                data_mean.to_csv(
                    f'../data/regularization_factor/metrics/夏季数据（七月）平均/{data_name}_{missing_rate}average.csv',
                    encoding='utf-8-sig')
            elif data_name == 'shiyue':
                data_mean.to_csv(
                    f'../data/regularization_factor/metrics/秋季数据（十月）平均/{data_name}_{missing_rate}average.csv',
                    encoding='utf-8-sig')
        data_month_mean = data_month[0].copy()
        for j in range(1, len(data_month)):
            data_month_mean = pd.concat([data_month_mean, data_month[j]], axis=1)
        if data_name == 'yiyue':
            data_month_mean.to_csv(
                f'../data/regularization_factor/metrics/冬季数据（一月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'siyue':
            data_month_mean.to_csv(
                f'../data/regularization_factor/metrics/春季数据（四月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'qiyue':
            data_month_mean.to_csv(
                f'../data/regularization_factor/metrics/夏季数据（七月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')
        elif data_name == 'shiyue':
            data_month_mean.to_csv(
                f'../data/regularization_factor/metrics/秋季数据（十月）平均/{data_name}_average_all.csv',
                encoding='utf-8-sig')


def combine_data_ts_compare():
    for data_name in ['yiyue_1800', 'siyue_1800', 'qiyue_1800', 'shiyue_1800']:
        data = []
        f = []
        numble_list = []
        mode_list = []
        missing_rates = [0.1, 0.2, 0.3, 0.4]
        for missing_rate in missing_rates:
            folder_path = f'Discussion/ts_compare/metrics/{missing_rate}missing_rate/{data_name}'
            files = os.listdir(folder_path)
            for file in files:
                if not os.path.isdir(os.path.join(folder_path, file)):
                    path = os.path.join(folder_path, file)
                    df = pd.read_csv(path)

                    # 为当前缺失率的列加前缀
                    df.columns = [f"{col}_{missing_rate}" for col in df.columns]

                    data.append(df)
                    name, ext = os.path.splitext(file)
                    f.append(name)
                    print(file)
            print('-------------------------')
        # 拆分数据名称
        for name in f:
            mode_name = name.split('_')[0]
            number_name = name.split('_')[1]
            numble_list.append(number_name)
            mode_list.append(mode_name)

        # 模型名称去重
        mode_list_update = []
        for item in mode_list:
            # 如果元素不在新列表中，则添加到新列表中
            if item not in mode_list_update:
                mode_list_update.append(item)

        # 将元素以4个为一组，组成新的列表
        q = []
        qq = []
        # data_list_update = [data[i:i + 4] for i in range(0, len(data), 4)]
        data_list_update = [[data[i + j * 4] for j in range(len(data) // 4)] for i in range(4)]
        for i in range(len(data_list_update[0])):
            q.clear()
            for j in range(len(data_list_update)):
                p = data_list_update[j][i].copy()
                q.append(p)
            qq.append(copy.deepcopy(q))
        pp = []
        for i in range(len(qq)):
            df = pd.concat(qq[i], ignore_index=True)
            pp.append(df)
        combined_df = pd.concat(pp, axis=1)
        combined_df.index = mode_list_update
        # 保存数据
        combined_df.to_csv(f'Discussion/ts_compare/metrics/{data_name}_all.csv')


if __name__ == '__main__':
    combine_data_ts_compare()
