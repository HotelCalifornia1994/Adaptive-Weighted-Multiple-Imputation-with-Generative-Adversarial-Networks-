import os


def modify_files_except_a_csv(folder_path):
    try:
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)

        # 创建一个字典来映射旧文件名到新文件名
        rename_map = {
            'regain1.csv': 'regain_1.csv',
            'regain2.csv': 'regain_2.csv',
            'regain3.csv': 'regain_3.csv'
        }

        # 遍历文件夹中的所有文件
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)

            # 如果文件名在要修改的列表中
            if file in rename_map:
                new_file_path = os.path.join(folder_path, rename_map[file])
                os.rename(file_path, new_file_path)

        print("指定文件名称已修改")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    # 调用函数并传入文件夹路径
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            folder_path = f"metrics\\{missing_rate}missing_rate/{data_name}"
            modify_files_except_a_csv(folder_path)
