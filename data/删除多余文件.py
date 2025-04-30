import os


def delete_files_except_a_csv(folder_path):
    try:
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)

        # 遍历文件夹中的所有文件
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)

            # 如果文件是/不是___.csv并且是一个普通文件，则删除
            if file in ['remgain_1.csv', 'remgain_2.csv', 'remgain_3.csv'] and os.path.isfile(file_path):
                os.remove(file_path)
                print(f"文件 '{file}' 已被删除")

        print("除了指定文件之外的所有数据文件已被删除")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    # 调用函数并传入文件夹路径
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path1 = f"metrics\\{missing_rate}missing_rate/{data_name}"
            folder_path2 = f"imputation_data\\{missing_rate}missing_rate/{data_name}"

            delete_files_except_a_csv(folder_path1)
            delete_files_except_a_csv(folder_path2)
