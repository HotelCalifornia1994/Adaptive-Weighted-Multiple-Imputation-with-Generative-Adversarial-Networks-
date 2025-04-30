import pandas as pd


# 调整数据排布
def rearrange_form():
    season_mapping = {
        '春季数据（四月）': 'siyue',
        '夏季数据（七月）': 'qiyue',
        '秋季数据（十月）': 'shiyue',
        '冬季数据（一月）': 'yiyue'}
    for file_name in ['春季数据（四月）', '夏季数据（七月）', '秋季数据（十月）', '冬季数据（一月）']:
        data_name = season_mapping[file_name]
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            file_path = f'metrics/{file_name}/{data_name}_{missing_rate}all.csv'
            df = pd.read_csv(file_path)
            df_len = len(df)
            combined_list = list([] for _ in range(df_len))
            for i in range(0, df_len):
                df_name_temp = df.iloc[i, 0]
                df_data_temp = df.iloc[i, :].values.tolist()
                if df_name_temp == 'gain(+l2+自编码位置编码+多重)':
                    combined_list[7] = df_data_temp
                elif df_name_temp == 'em':
                    combined_list[0] = df_data_temp
                elif df_name_temp == 'mean':
                    combined_list[1] = df_data_temp
                elif df_name_temp == 'median':
                    combined_list[2] = df_data_temp
                elif df_name_temp == 'locf':
                    combined_list[3] = df_data_temp
                elif df_name_temp == 'knn':
                    combined_list[4] = df_data_temp
                elif df_name_temp == 'mice':
                    combined_list[5] = df_data_temp
                elif df_name_temp == 'missforest':
                    combined_list[6] = df_data_temp
            df_update = pd.DataFrame(combined_list, columns=df.columns)
            df_update.to_csv(file_path, encoding='utf-8-sig', index=False)


def rearrange_ablation_form():
    season_mapping = {
        '春季数据（四月）': 'siyue',
        '夏季数据（七月）': 'qiyue',
        '秋季数据（十月）': 'shiyue',
        '冬季数据（一月）': 'yiyue'}
    for file_name in ['春季数据（四月）', '夏季数据（七月）', '秋季数据（十月）', '冬季数据（一月）']:
        data_name = season_mapping[file_name]
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            file_path = f'Ablation/metrics/{file_name}/{data_name}_{missing_rate}all.csv'
            df = pd.read_csv(file_path)
            df_len = len(df)
            combined_list = list([] for _ in range(df_len))
            for i in range(0, df_len):
                df_name_temp = df.iloc[i, 0]
                df_data_temp = df.iloc[i, :].values.tolist()
                if df_name_temp == 'gain(原始)':
                    combined_list[0] = df_data_temp
                elif df_name_temp == 'gain(+l2)':
                    combined_list[1] = df_data_temp
                elif df_name_temp == 'gain(+自编码位置编码)':
                    combined_list[2] = df_data_temp
                elif df_name_temp == 'gain(+多重)':
                    combined_list[3] = df_data_temp
                elif df_name_temp == 'gain(+l2+自编码位置编码)':
                    combined_list[4] = df_data_temp
                elif df_name_temp == 'gain(+l2+自编码位置编码+多重)':
                    combined_list[5] = df_data_temp
            df_update = pd.DataFrame(combined_list, columns=df.columns)
            df_update.to_csv(file_path, encoding='utf-8-sig', index=False)


def rearrange_form_average():
    season_mapping = {
        '春季数据（四月）平均': 'siyue',
        '夏季数据（七月）平均': 'qiyue',
        '秋季数据（十月）平均': 'shiyue',
        '冬季数据（一月）平均': 'yiyue'}
    for file_name in ['春季数据（四月）平均', '夏季数据（七月）平均', '秋季数据（十月）平均', '冬季数据（一月）平均']:
        data_name = season_mapping[file_name]
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            file_path = f'metrics/{file_name}/{data_name}_{missing_rate}average.csv'
            df = pd.read_csv(file_path)
            df_len = len(df)
            combined_list = list([] for _ in range(df_len))
            for i in range(0, df_len):
                df_name_temp = df.iloc[i, 0]
                df_data_temp = df.iloc[i, :].values.tolist()
                if df_name_temp == 'gain(+l2+自编码位置编码+多重)':
                    combined_list[7] = df_data_temp
                elif df_name_temp == 'em':
                    combined_list[0] = df_data_temp
                elif df_name_temp == 'mean':
                    combined_list[1] = df_data_temp
                elif df_name_temp == 'median':
                    combined_list[2] = df_data_temp
                elif df_name_temp == 'locf':
                    combined_list[3] = df_data_temp
                elif df_name_temp == 'knn':
                    combined_list[4] = df_data_temp
                elif df_name_temp == 'mice':
                    combined_list[5] = df_data_temp
                elif df_name_temp == 'missforest':
                    combined_list[6] = df_data_temp
            df_update = pd.DataFrame(combined_list, columns=df.columns)
            df_update.to_csv(file_path, encoding='utf-8-sig', index=False)


def rearrange_form_average_all():
    season_mapping = {
        '春季数据（四月）平均': 'siyue',
        '夏季数据（七月）平均': 'qiyue',
        '秋季数据（十月）平均': 'shiyue',
        '冬季数据（一月）平均': 'yiyue'}
    for file_name in ['春季数据（四月）平均', '夏季数据（七月）平均', '秋季数据（十月）平均', '冬季数据（一月）平均']:
        data_name = season_mapping[file_name]
        file_path = f'Ablation/metrics/{file_name}/{data_name}_average_all.csv'
        df = pd.read_csv(file_path)
        df_len = len(df)
        combined_list = list([] for _ in range(df_len))
        for i in range(0, df_len):
            df_name_temp = df.iloc[i, 0]
            df_data_temp = df.iloc[i, :].values.tolist()
            if df_name_temp == 'gain(原始)':
                combined_list[0] = df_data_temp
            elif df_name_temp == 'gain(+l2)':
                combined_list[1] = df_data_temp
            elif df_name_temp == 'gain(+自编码位置编码)':
                combined_list[2] = df_data_temp
            elif df_name_temp == 'gain(+多重)':
                combined_list[3] = df_data_temp
            elif df_name_temp == 'gain(+l2+自编码位置编码)':
                combined_list[4] = df_data_temp
            elif df_name_temp == 'gain(+l2+自编码位置编码+多重)':
                combined_list[5] = df_data_temp
            elif df_name_temp == 'gain(+l1+自编码位置编码)':
                combined_list[6] = df_data_temp
            elif df_name_temp == 'gain(+droupout+自编码位置编码)':
                combined_list[7] = df_data_temp
        df_update = pd.DataFrame(combined_list, columns=df.columns)
        df_update.to_csv(file_path, encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    rearrange_form_average_all()
