import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from other.data_loader import data_loader
from cycler import cycler
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.ticker import AutoLocator
import matplotlib.patheffects as path_effects
from scipy.stats import friedmanchisquare
import seaborn as sns
from utils import r2_score
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import ks_2samp


# 绘制缺失数据图
def missing_data_plot():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in [1, 2, 3]:
                plt.rc("font", family='DengXian')  # 解决中文乱码
                ori_data, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                mask = data_m == 1
                ori_data_x_df = pd.DataFrame(ori_data)
                miss_data_x = ori_data_x_df.where(~mask, other=np.nan)
                ori_data_x = ori_data_x_df.where(mask, other=np.nan)
                # miss_data_x.columns = ['WTG1', 'WTG2', 'WTG3', 'WTG4', 'WTG5']
                miss_data_x.columns = ['风机1', '风机2', '风机3', '风机4', '风机5']
                # ori_data_x.columns = ['WTG1', 'WTG2', 'WTG3', 'WTG4', 'WTG5']
                ori_data_x.columns = ['风机1', '风机2', '风机3', '风机4', '风机5']
                colors = ['tomato', 'orange', 'limegreen', 'dodgerblue', 'mediumpurple']
                # 图形调整
                fig, axes = plt.subplots(nrows=len(ori_data_x.columns), ncols=1, figsize=(20, 12))
                # Define the number of ticks
                num_ticks_x = 8  # Adjust the number of ticks for the x-axis
                num_ticks_y = 5  # Adjust the number of ticks for the y-axis
                # for ax in axes:
                for i, (col, ax) in enumerate(zip(ori_data_x.columns, axes)):
                    ori_data_x[col].plot(ax=ax, legend=True, color=colors[i])
                    ax.tick_params(axis='both', which='major', labelsize=20, width=2, labelcolor='black',
                                   colors='black')
                    ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 30})  # 加粗图例字体
                    # 控制 x 轴和 y 轴的主刻度（major ticks）的数量和分布
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks_x))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks_y))
                    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
                    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
                    ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
                    ax.spines['top'].set_linewidth(2)  # 设置上部坐标轴的粗细
                    # 设置刻度字体加粗
                    # ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=20)
                    # ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=20)
                    # 设置刻度字体加粗
                    xticks = ax.get_xticks()
                    ax.set_xticklabels([f'{x:.1f}' for x in xticks], fontweight='bold', fontsize=38)
                    yticks = ax.get_yticks()
                    ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontweight='bold', fontsize=38)

                    # 强制将 Y 轴和 X 轴刻度标签格式化为整数
                    def format_func(value, tick_number):
                        return f'{int(value)}'  # 将刻度值格式化为整数

                    ax.yaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 Y 轴
                    ax.xaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 x 轴
                    # 设置轴标签
                    ax.set_xlabel('时间点(10分钟间隔)', fontsize=50, fontweight='bold')  # 设置 x 轴标签为“样本点”
                    # ax.set_ylabel('风速', fontsize=20, fontweight='bold')  # 设置 y 轴标签为“风速”
                    # 只在第一个子图设置 y 轴标签
                    # if i == 0:
                    #     ax.set_ylabel('风速', fontsize=20, fontweight='bold')  # 设置 y 轴标签为“风速”

                    # 隐藏所有子图的 y 轴标签
                    ax.set_ylabel('', fontsize=50, fontweight='bold')
                # 在中间子图的位置添加 y 轴标签
                middle_index = len(axes) // 2  # 找到中间子图的索引
                axes[middle_index].set_ylabel('风速（m/s）', fontsize=50, fontweight='bold')  # 设置 y 轴标签为“风速”

                plt.tight_layout(h_pad=1)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
                # plt.gca().xaxis.label.set_fontweight('bold')
                # plt.gca().yaxis.label.set_fontweight('bold')

                # 存储图形
                # save_path = f'plots/missingdata/{missing_rate}missing_rate/{data_name}/missingdata_{random_seed}.png'
                save_path = f'plots/missingdata_cn/{missing_rate}missing_rate/{data_name}/missingdata_{random_seed}.eps'
                plt.savefig(save_path)
                # plt.show()
                plt.close()
                print(f'\033[31m{data_name}数据集，{missing_rate}的缺失率，第{random_seed}个图形已保存\033[0m')


# 绘制完整数据图
def complete_data_plot():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        data_x = np.loadtxt(f'months/{data_name}.csv', delimiter=',')
        data_x = pd.DataFrame(data_x)
        # data_x.columns = ['WTG1', 'WTG2', 'WTG3', 'WTG4', 'WTG5']
        data_x.columns = ['风机1', '风机2', '风机3', '风机4', '风机5']
        colors = ['tomato', 'orange', 'limegreen', 'dodgerblue', 'mediumpurple']

        # 图形调整
        plt.rc("font", family='DengXian')  # 解决中文乱码
        fig, axes = plt.subplots(nrows=len(data_x.columns), ncols=1, figsize=(20, 12))
        # ax = data_x.plot(legend=True, subplots=True, figsize=(20, 10), color=colors)
        # x_ticks = np.arange(0, len(data_x), step=10)  # Example: every 10 units
        # y_ticks = np.arange(0, data_x.max().max(), step=2)  # Example: every 2 units, adjust as necessary

        # Define the number of ticks
        num_ticks_x = 8  # Adjust the number of ticks for the x-axis
        num_ticks_y = 5  # Adjust the number of ticks for the y-axis
        # 将图例放在右上角
        for i, (col, ax) in enumerate(zip(data_x.columns, axes)):
            data_x[col].plot(ax=ax, legend=True, color=colors[i])
            ax.tick_params(axis='both', which='major', labelsize=20, width=2, labelcolor='black',
                           colors='black')
            ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 30})  # 加粗图例字体
            # 控制 x 轴和 y 轴的主刻度（major ticks）的数量和分布
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks_x))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks_y))
            ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
            ax.spines['top'].set_linewidth(2)  # 设置上部坐标轴的粗细
            # 设置刻度字体加粗
            # ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=20)
            # ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=20)
            # 设置刻度字体加粗
            xticks = ax.get_xticks()
            ax.set_xticklabels([f'{x:.1f}' for x in xticks], fontweight='bold', fontsize=38)
            yticks = ax.get_yticks()
            ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontweight='bold', fontsize=38)

            # 强制将 Y 轴和 X 轴刻度标签格式化为整数
            def format_func(value, tick_number):
                return f'{int(value)}'  # 将刻度值格式化为整数

            ax.yaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 Y 轴
            ax.xaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 x 轴
            # 设置轴标签
            ax.set_xlabel('时间点(10分钟间隔)', fontsize=50, fontweight='bold')  # 设置 x 轴标签为“样本点”
            # ax.set_ylabel('风速', fontsize=20, fontweight='bold')  # 设置 y 轴标签为“风速”
            # 只在第一个子图设置 y 轴标签
            # if i == 0:
            #     ax.set_ylabel('风速', fontsize=20, fontweight='bold')  # 设置 y 轴标签为“风速”

            # 隐藏所有子图的 y 轴标签
            ax.set_ylabel('', fontsize=50, fontweight='bold')
            # 在中间子图的位置添加 y 轴标签
        middle_index = len(axes) // 2  # 找到中间子图的索引
        axes[middle_index].set_ylabel('风速（m/s）', fontsize=50, fontweight='bold')  # 设置 y 轴标签为“风速”
        plt.tight_layout(h_pad=1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        # plt.gca().xaxis.label.set_fontweight('bold')
        # plt.gca().yaxis.label.set_fontweight('bold')
        # 存储图形
        # save_path = f'plots/completedata/{data_name}.eps'
        save_path = f'plots/completedata_cn/{data_name}.eps'
        plt.savefig(save_path)
        # plt.show()
        plt.close()
        print(f'\033[31m{data_name}数据的图形已保存\033[0m')


# 绘制完整3D数据图
def complete_data_3dplot():
    # 读取数据
    for data_name in tqdm(['yiyue', 'siyue', 'qiyue', 'shiyue']):
        data_x = np.loadtxt(f'months/{data_name}.csv', delimiter=',')
        data_x = pd.DataFrame(data_x)
        # data_x.columns = ['WTG1', 'WTG2', 'WTG3', 'WTG4', 'WTG5']
        data_x.columns = ['风机1', '风机2', '风机3', '风机4', '风机5']
        # 定义颜色列表
        plt.rc("font", family='DengXian')  # 解决中文乱码
        colors = ['tomato', 'orange', 'limegreen', 'dodgerblue', 'mediumpurple']
        # 创建 3D 图形
        fig = plt.figure(figsize=(18, 9))
        # # ax = fig.add_subplot(111, projection='3d')
        # ax = fig.add_axes(Axes3D(fig))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1.1, 0.5, 1]))  # xyz轴的长宽高
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=30, azim=40)  # elev=20将观察者从z轴正方向向下看20度，azim然后逆时针旋转30度
        ax.xaxis.labelpad = 100  # 标签间距

        # 加粗刻度线和刻度标签
        ax.tick_params(axis='both', which='major', width=10, labelsize=12)

        # 调整布局
        # plt.tight_layout(h_pad=1)
        # plt.subplots_adjust(left=0.05, right=0.8, bottom=0.05, top=0.9)
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # 绘制图形
        # 迭代每一列数据并绘制到3D图中
        for i, col in enumerate(data_x.columns):
            # 生成 Z 轴的位置，避免数据重叠
            zs = np.full_like(data_x[col], i)
            ax.plot(zs, data_x.index, data_x[col], label=col, color=colors[i % len(colors)])

        # 设置标签和图例
        # ax.set_xlabel('Column')
        # ax.set_ylabel('Time(10min)', fontsize=12)
        ax.set_ylabel('时间点(10分钟间隔)', fontsize=18, fontweight='bold')
        # ax.set_zlabel('Wind speeds(m/s)', fontsize=12)
        ax.set_zlabel('风速(m/s)', fontsize=18, fontweight='bold')

        # 设置 X 轴刻度和标签
        ax.set_xticks(np.arange(len(data_x.columns)))
        ax.set_xticklabels(data_x.columns, fontsize=15, fontweight='bold')
        ax.set_xlim(0, len(data_x.columns) - 1)  # X 轴范围
        ax.set_ylim(0, len(data_x.index) - 1)  # Y 轴范围
        ax.set_zlim(data_x.values.min(), data_x.values.max())  # Z 轴范围
        ax.set_yticklabels(ax.get_yticks(), fontsize=15, fontweight='bold')  # 加粗 Y 轴刻度标签
        ax.set_zticklabels(ax.get_zticks(), fontsize=15, fontweight='bold')  # 加粗 Z 轴刻度标签
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 确保 Y 轴刻度为整数
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))  # 确保 Z 轴刻度为整数
        ax.yaxis.labelpad = 15  # 增加 ylabel 与刻度线之间的距离

        # ax.tick_params(axis='both', which='major', width=2, labelsize=12)
        # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        # plt.gca().tick_params(axis='both', which='major', labelsize=15, width=10)
        # 强制将 Y 轴和 Z 轴刻度标签格式化为整数
        def format_func(value, tick_number):
            return f'{int(value)}'  # 将刻度值格式化为整数

        ax.yaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 Y 轴
        ax.zaxis.set_major_formatter(FuncFormatter(format_func))  # 应用格式化到 Z 轴

        # 设置图例位置和样式
        # plt.show()
        # 存储图形
        # save_path = f'plots/completedata/{data_name}_3d.eps'
        save_path = f'plots/completedata_cn/{data_name}_3d.eps'
        plt.savefig(save_path)
        plt.close()  # 关闭当前图形，防止重叠

        print(f'\033[31m{data_name}数据的所有列的3D图形已保存\033[0m')


# 对比模型
def compare_plot():
    season_mapping = {
        'siyue': '春季数据（四月）',
        'qiyue': '夏季数据（七月）',
        'shiyue': '秋季数据（十月）',
        'yiyue': '冬季数据（一月）'}
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in [1, 2, 3]:
                # for data_name in ['yiyue']:
                #     for missing_rate in [0.6]:
                #         for random_seed in [1]:
                for column in [0, 1, 2, 3, 4]:
                    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                    em = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\em_{random_seed}.csv',
                        delimiter=',')
                    knn = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\knn_{random_seed}.csv',
                        delimiter=',')
                    mean = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\mean_{random_seed}.csv',
                        delimiter=',')
                    median = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\median_{random_seed}.csv',
                        delimiter=',')
                    mice = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\mice_{random_seed}.csv',
                        delimiter=',')
                    missforest = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\missforest_{random_seed}.csv',
                        delimiter=',')
                    proposed_model = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                        delimiter=',')
                    em_1 = em[500:1000, column]
                    knn_1 = knn[500:1000, column]
                    mean_1 = mean[500:1000, column]
                    median_1 = median[500:1000, column]
                    mice_1 = mice[500:1000, column]
                    missforest_1 = missforest[500:1000, column]
                    proposed_model_1 = proposed_model[500:1000, column]
                    ori_data_x_1 = ori_data_x[500:1000, column]
                    missing_data_1 = missing_data[500:1000, column]
                    data_m_1 = data_m[500:1000, column]
                    mask = data_m_1 == 1
                    time = np.arange(0, len(ori_data_x_1), 1)
                    month = season_mapping[data_name]

                    # 绘制折线图
                    plt.figure(figsize=(15, 5))
                    m, n = 1, 1
                    mark = 0
                    for i in range(len(ori_data_x_1) - 1):
                        if mask[i] and mask[i + 1]:
                            plt.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='black', marker='',
                                     markersize=3, alpha=1, zorder=10)
                        elif not mask[i] and not mask[i + 1]:
                            plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='blue', marker='',
                                     alpha=0.6,
                                     markersize=2, label='missing data' if m == 1 else "", zorder=9)
                            plt.plot(time[i:i + 2], em_1[i:i + 2], linestyle='-.', color='green', alpha=0.5,
                                     label='em' if m == 1 else "")
                            plt.plot(time[i:i + 2], knn_1[i:i + 2], linestyle='-.', color='orange', alpha=0.5,
                                     label='knn' if m == 1 else "")
                            plt.plot(time[i:i + 2], mean_1[i:i + 2], linestyle='-.', color='purple', alpha=0.5,
                                     label='mean' if m == 1 else "")
                            plt.plot(time[i:i + 2], median_1[i:i + 2], linestyle='-.', color='pink', alpha=0.5,
                                     label='median' if m == 1 else "")
                            plt.plot(time[i:i + 2], mice_1[i:i + 2], linestyle='-.', color='olive', alpha=0.5,
                                     label='mice' if m == 1 else "")
                            plt.plot(time[i:i + 2], missforest_1[i:i + 2], linestyle='-.', color='sienna', alpha=0.5,
                                     label='missforest' if m == 1 else "")
                            plt.plot(time[i:i + 2], proposed_model_1[i:i + 2], linestyle='-', color='red', marker='.',
                                     markersize=3,
                                     label='proposed model' if m == 1 else "", zorder=10)
                            m = 0
                            if mark == 0:
                                mark = 1
                        else:
                            plt.plot([time[i], time[i + 1]],
                                     [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                                      ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                                     marker='.', markersize=3, alpha=1,
                                     color='black', label='original data' if n == 1 else "", zorder=10)
                            n = 0
                            if mark == 0:
                                mark = 2
                    global order
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if mark == 1:
                        order = [7, 8, 0, 1, 2, 3, 4, 5, 6]  # Reorder legend labels
                    elif mark == 2:
                        order = [8, 0, 1, 2, 3, 5, 5, 6, 7]
                    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                               loc='upper right', ncol=2)
                    plt.gca().tick_params(axis='both', which='major', labelsize=12)
                    # plt.title('Line Plot with Missing Data and Interpolation')
                    plt.xlabel('Time(10min)', fontsize=12)
                    plt.ylabel('Wind speeds(m/s)', fontsize=12)
                    plt.tight_layout()
                    # plt.show()
                    # 存储图形
                    save_path = f'plots/compare/{missing_rate}missing_rate/{data_name}/{random_seed}/{month}{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比实验图.png'
                    plt.savefig(save_path)
                    plt.close()  # 关闭当前图形，防止重叠

                    print(
                        f'\033[31m{month}数据{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比实验图已保存\033[0m')


# 对比模型R2
def compare_plot_r2():
    # 定义模型列表
    models = ['gain(+l2+自编码位置编码+多重)', 'em', 'knn', 'mean', 'median', 'mice', 'missforest', 'locf']
    model_colors = ['#a51c36', '#4485c7', '#682487', '#7abbdb', '#84ba42', '#d4562e', '#dbb428', '#CE6090']
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.2, 0.5, 0.8]:
            for random_seed in [1]:
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                mask = data_m == 0
                # 读取每个模型的数据
                all_model_data = {}
                for i, model in enumerate(models):
                    file_path = f'../data/imputation_data/{missing_rate}missing_rate/{data_name}/{model}_{random_seed}.csv'
                    all_model_data[model] = pd.read_csv(file_path, header=None).values
                # 计算每个模型的 R² 值
                r2_scores = {}
                for model in models:
                    r2_scores[model] = r2_score(ori_data_x, all_model_data[model], data_m)

                # 绘制每个模型的散点图和 R² 图
                for model, color in zip(models, model_colors):
                    x = ori_data_x[mask]
                    model_y = all_model_data[model][mask]

                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, model_y, label=f'{model} (R² = {r2_scores[model]:.3f})',
                                marker='.', color=color, alpha=0.6, s=100)
                    # 添加 y = x 参考线
                    min_val = min(x.min(), model_y.min())
                    max_val = max(x.max(), model_y.max())
                    plt.plot([min_val, max_val], [min_val, max_val],
                             color='black', linestyle='--', linewidth=1.5, label='y = x')

                    # 在图右下角添加 R² 值
                    plt.text(x=max_val, y=min_val,
                             s=f'R² = {r2_scores[model]:.3f}',
                             ha='right', va='bottom',
                             fontsize=20,
                             fontweight='bold',
                             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

                    ax = plt.gca()  # 获取当前坐标轴
                    for spine in ['bottom', 'left', 'top', 'right']:
                        ax.spines[spine].set_linewidth(1.5)
                    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
                    plt.yticks(fontsize=20, fontweight='bold')
                    plt.xticks(fontsize=20, fontweight='bold')

                    plt.xlabel('True values', fontsize=25, fontweight='bold')
                    plt.ylabel('Imputation values', fontsize=25, fontweight='bold')

                    plt.tight_layout()
                    # plt.show()
                    # 存储图形
                    save_path = f'plots/compare_r2/{data_name}/{missing_rate}missing_rate/{model}_{missing_rate}缺失率_{random_seed}R2图.eps'
                    plt.savefig(save_path, dpi=300)


# 改进后的模型与原始曲线的对比
def imputation_plot():
    plt.rc("font", family='DengXian')  # 解决中文乱码
    season_mapping = {
        'siyue': '春季数据（四月）',
        'qiyue': '夏季数据（七月）',
        'shiyue': '秋季数据（十月）',
        'yiyue': '冬季数据（一月）'}
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in [1, 2, 3]:
                for column in [0, 1, 2, 3, 4]:
                    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                    imputation_data = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                        delimiter=',')
                    imputation_data_1 = imputation_data[:, column]
                    ori_data_x_1 = ori_data_x[:, column]
                    missing_data_1 = missing_data[:, column]
                    data_m_1 = data_m[:, column]
                    mask = data_m_1 == 1
                    time = np.arange(0, len(ori_data_x_1), 1)
                    month = season_mapping[data_name]

                    # 绘制折线图
                    plt.figure(figsize=(20, 12))
                    m, n = 1, 1
                    for i in range(len(ori_data_x_1) - 1):
                        if mask[i] and mask[i + 1]:
                            plt.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='#377EB9', marker='',
                                     zorder=10)
                        elif not mask[i] and not mask[i + 1]:
                            plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                                     markersize=2, label='Missing Data' if m == 1 else "")  # color='#198E5C'
                            plt.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                                     label='Imputed Data' if m == 1 else "", zorder=10)  # color='#EC050E'
                            # plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                            #          markersize=2, label='缺失数据' if m == 1 else "")  # color='#198E5C'
                            # plt.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                            #          label='插补数据' if m == 1 else "", zorder=10)  # color='#EC050E'
                            m = 0
                        else:
                            # plt.plot([time[i], time[i + 1]],
                            #          [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                            #           ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                            #          color='#377EB9', label='原始数据' if n == 1 else "", zorder=10)  # color='#5C8DC7'
                            plt.plot([time[i], time[i + 1]],
                                     [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                                      ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                                     color='#377EB9', label='Original Data' if n == 1 else "",
                                     zorder=10)  # color='#5C8DC7'
                            n = 0
                    # 获取当前图形的坐标轴对象
                    ax = plt.gca()
                    # 设置坐标轴边框线条加粗
                    ax.spines['top'].set_linewidth(1.5)  # 设置上边框加粗
                    ax.spines['right'].set_linewidth(1.5)  # 设置右边框加粗
                    ax.spines['bottom'].set_linewidth(1.5)  # 设置下边框加粗
                    ax.spines['left'].set_linewidth(1.5)  # 设置左边框加粗
                    ax.tick_params(axis='both', which='major', labelsize=40, width=2)  # 主刻度字体加粗，线条加粗
                    # 设置刻度字体加粗
                    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=40)
                    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=40)
                    plt.legend(prop={'weight': 'bold', 'size': 30})
                    # plt.title('Line Plot with Missing Data and Interpolation')
                    plt.xlabel('Time(10min)', fontsize=50, fontweight='bold')
                    # plt.xlabel('时间点（十分钟间隔）', fontsize=50, fontweight='bold')
                    plt.ylabel('Wind speeds(m/s)', fontsize=50, fontweight='bold')
                    # plt.ylabel('风速(m/s)', fontsize=50, fontweight='bold')

                    # 添加放大镜效果
                    # 选择要放大的区域（这里假设放大中间1/3的区域）
                    x1, x2 = len(ori_data_x_1) // 10, 2 * len(ori_data_x_1) // 10
                    y1 = min(ori_data_x_1[x1:x2]) - 0.5
                    y2 = max(ori_data_x_1[x1:x2]) + 0.5
                    # 创建放大镜子图
                    axins = ax.inset_axes((0.695, 0.69, 0.3, 0.3))  # 放大镜位置和大小
                    # **加粗放大镜的边框**
                    for spine in axins.spines.values():
                        spine.set_linewidth(1)  # 可以调整数值来改变粗细
                    for i in range(len(ori_data_x_1) - 1):
                        if mask[i] and mask[i + 1] and x1 <= i <= x2:
                            axins.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='#377EB9',
                                       marker='', zorder=10)
                        elif not mask[i] and not mask[i + 1] and x1 <= i <= x2:
                            axins.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                                       markersize=2)
                            axins.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                                       zorder=10)
                        elif x1 <= i <= x2:
                            axins.plot([time[i], time[i + 1]],
                                       [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                                        ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]],
                                       linestyle='-', color='#377EB9', zorder=10)

                    # 设置放大镜的坐标范围
                    axins.set_xlim(time[x1], time[x2 - 1])
                    axins.set_ylim(y1, y2)
                    axins.set_xticklabels([])
                    axins.set_yticklabels([])
                    axins.grid(True, linestyle='--', alpha=0.7)

                    # 在主图中标记放大区域
                    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, linewidth=1)

                    # 在主图中添加放大区域的矩形标记
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='black', facecolor='none',
                                         linestyle='--')
                    ax.add_patch(rect)
                    # 连接放大镜区域与主图
                    # rect = plt.Rectangle((time[x1], y1), time[x2 - 1] - time[x1], y2 - y1, linewidth=1.5,
                    #                      edgecolor='black', facecolor='none', linestyle='--')
                    # ax.add_patch(rect)
                    plt.tight_layout()
                    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    plt.tight_layout(pad=2.0)
                    # plt.show()
                    # 存储图形
                    save_path = f'plots/imputationdata/{missing_rate}missing_rate/{data_name}/{random_seed}/{month}{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比图.eps'
                    # save_path = f'plots/imputationdata_cn/{missing_rate}missing_rate/{data_name}/{random_seed}/{month}{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比图.eps'
                    plt.savefig(save_path, dpi=300)
                    plt.close()  # 关闭当前图形，防止重叠

                    print(
                        f'\033[31m{month}数据{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比图已保存\033[0m')


# 消融实验（数据曲线）
def ablation_plot():
    season_mapping = {
        'siyue': '春季数据（四月）',
        'qiyue': '夏季数据（七月）',
        'shiyue': '秋季数据（十月）',
        'yiyue': '冬季数据（一月）'}
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in [1, 2, 3]:
                # for data_name in ['shiyue']:
                #     for missing_rate in [0.2]:
                #         for random_seed in [1]:
                for column in [0, 1, 2, 3, 4]:
                    ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                    imputation_data = np.loadtxt(
                        f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                        delimiter=',')
                    imputation_data_1 = imputation_data[:, column]
                    ori_data_x_1 = ori_data_x[:, column]
                    missing_data_1 = missing_data[:, column]
                    data_m_1 = data_m[:, column]
                    mask = data_m_1 == 1
                    time = np.arange(0, len(ori_data_x_1), 1)
                    month = season_mapping[data_name]

                    # 绘制折线图
                    ax = plt.figure(figsize=(15, 5))
                    m, n = 1, 1
                    for i in range(len(ori_data_x_1) - 1):
                        if mask[i] and mask[i + 1]:
                            plt.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='#5C8DC7', marker='',
                                     zorder=10)
                        elif not mask[i] and not mask[i + 1]:
                            plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                                     markersize=2, label='Missing data' if m == 1 else "")
                            plt.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                                     label='Imputation data' if m == 1 else "", zorder=10)
                            m = 0
                        else:
                            plt.plot([time[i], time[i + 1]],
                                     [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                                      ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                                     color='#5C8DC7', label='Original data' if n == 1 else "", zorder=10)
                            n = 0

                    # plt.yticks(fontsize=20)
                    # plt.tick_params(labelsize=10)

                    plt.legend()
                    # plt.title('Line Plot with Missing Data and Interpolation')
                    plt.gca().tick_params(axis='both', which='major', labelsize=12)
                    plt.xlabel('Time(10min)', fontsize=12)
                    plt.ylabel('Wind speeds(m/s)', fontsize=12)
                    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    plt.tight_layout()
                    plt.show()
                    # 存储图形
                    save_path = f'plots/Ablation/{missing_rate}missing_rate/{data_name}/{random_seed}/{month}{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的消融实验对比图.png'
                    # plt.savefig(save_path)
                    plt.close()  # 关闭当前图形，防止重叠

                    print(
                        f'\033[31m{month}数据{missing_rate}缺失率第{random_seed}个样本的第{column + 1}个风机的插补对比图已保存\033[0m')


# 正则化对比实验
def radar_chart():
    plt.rc("font", family='DengXian')  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False  # 负号报错
    season_mapping = {
        '春季数据（四月）平均': 'siyue',
        '夏季数据（七月）平均': 'qiyue',
        '秋季数据（十月）平均': 'shiyue',
        '冬季数据（一月）平均': 'yiyue'}
    for file_name in ['春季数据（四月）平均', '夏季数据（七月）平均', '秋季数据（十月）平均', '冬季数据（一月）平均']:
        data_name = season_mapping[file_name]
        data = pd.read_csv(
            open(f'D:/Desktop/我的插补/data/Ablation/metrics/{file_name}/{data_name}_average_all.csv',
                 encoding='utf-8'))
        data.set_index(data.columns[0], inplace=True)

        rmse_concat = ['rmse', 'rmse.1', 'rmse.2', 'rmse.3', 'rmse.4', 'rmse.5', 'rmse.6', 'rmse.7']
        x = pd.Series(
            ['0.1missing rate', '0.2missing rate', '0.3missing rate', '0.4missing rate', '0.5missing rate',
             '0.6missing rate',
             '0.7missing rate', '0.8missing rate'])
        # x = pd.Series(
        #     ['0.1缺失率', '0.2缺失率', '0.3缺失率', '0.4缺失率', '0.5缺失率',
        #      '0.6缺失率',
        #      '0.7缺失率', '0.8缺失率'])
        l1 = data.loc['gain(+l1+自编码位置编码)', rmse_concat]
        l2 = data.loc['gain(+l2+自编码位置编码)', rmse_concat]
        droupout = data.loc['gain(+droupout+自编码位置编码)', rmse_concat]
        N = len(x)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        #    0
        #  /   \
        # 3     1    [0,1,2,3,0]才能画出一个首尾相连的多边形，一共N+1个点
        #  \   /
        #    2
        theta = np.concatenate((theta, [theta[0]]), axis=0)  # 默认axis=0
        l1 = np.concatenate((l1, [l1[0]]))
        l2 = np.concatenate((l2, [l2[0]]))
        droupout = np.concatenate((droupout, [droupout[0]]))
        x = np.concatenate((x, [x[0]]))

        # 多边形雷达图
        # 隐藏默认圆形框线，手动绘制多边形等间距框线
        plt.figure(figsize=(14, 12))
        rad = plt.subplot(111, polar=True)
        # 画形状相同，顶点间等距的多边形
        # 0，20，40，60，80，100
        if data_name == 'yiyue':
            for i in np.arange(0, 2.5 + 0.5, 0.5):
                rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
            # 用直线连接起这些多边形同一角度的顶点
            for i in range(N):
                rad.plot([theta[i], theta[i]], [0, 2.5],
                         '-.', lw=1, color='black', alpha=0.4)
        else:
            for i in np.arange(0, 2 + 0.4, 0.4):
                rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
            # 用直线连接起这些多边形同一角度的顶点
            for i in range(N):
                rad.plot([theta[i], theta[i]], [0, 2],
                         '-.', lw=1, color='black', alpha=0.4)
        # 隐藏黑色圆边界线
        rad.spines['polar'].set_visible(False)
        # 隐藏默认圆形网格线
        rad.grid(False)
        rad.set_yticklabels([])  # 隐藏径向刻度标签

        rad.fill(theta, l2, color='r', alpha=0.25)

        rad.plot(theta, l2, color='r', alpha=0.5, linewidth=1.5,
                 label='GAIN with L2 penalty',
                 linestyle='-', marker='p', markersize=8)
        rad.plot(theta, l1, color='g', alpha=0.5, linewidth=1.5,
                 label='GAIN with L1 penalty',
                 linestyle='-', marker='o', markersize=8)
        rad.plot(theta, droupout, color='b', alpha=0.5, linewidth=1.5,
                 label='GAIN with Dropout',
                 linestyle='-', marker='o', markersize=8)
        # 标注每个点的值
        for i, m in zip(theta, l1):
            txt = rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=25, color='g')
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='g')])
        for i, m in zip(theta, l2):
            txt = rad.text(i, m - 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='r')
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='r')])
        for i, m in zip(theta, droupout):
            txt = rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=25, color='b')
            txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='b')])
        # rad.set_thetagrids(theta * 180 / npy.pi, x)
        rad.set_thetagrids(theta[:-1] * 180 / np.pi, x[:-1])

        # 设置角度刻度标签，避免与坐标轴重叠
        # 通过调整 theta 和 x 的位置，确保标签不与坐标轴重叠
        rad.set_xticks(theta[:-1])  # 设置刻度位置
        rad.set_xticklabels(x[:-1], fontsize=35, fontweight='bold')  # 设置刻度标签
        for label in rad.get_xticklabels():
            label.set_y(label.get_position()[1] - 0.4)  # 负值让标签向外移动

        rad.set_theta_zero_location('S')  # 设置0刻度位置
        if data_name == 'yiyue':
            rad.set_rlim(0, 2.5)  # 设置坐标刻度范围
        else:
            rad.set_rlim(0, 2)
        rad.set_rlabel_position(60)  # 设置坐标显示角度
        # plt.gca().xaxis.label.set_fontweight('bold')
        # plt.gca().yaxis.label.set_fontweight('bold')
        # rad.set_title("RMSE of Regularization-Based Comparative Analysis Experiments", pad=60, fontweight='bold', fontsize=22)  # 设置标题
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, prop={'weight': 'bold', 'size': 25})
        # plt.legend(loc='best', prop={'weight': 'bold', 'size': 16})
        # plt.show()
        # 存储图形
        save_path = f'plots/Ablation/{data_name}正则化的对比分析实验雷达图.pdf'
        # save_path = f'plots/Ablation_cn/{data_name}正则化的对比分析实验雷达图.pdf'
        plt.savefig(save_path, dpi=300)
        plt.close()  # 关闭当前图形，防止重叠


# 正则化系数雷达图
def radar_regularization():
    plt.rc("font", family='DengXian')  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False  # 负号报错
    season_mapping = {
        '春季数据（四月）平均': 'siyue',
        '夏季数据（七月）平均': 'qiyue',
        '秋季数据（十月）平均': 'shiyue',
        '冬季数据（一月）平均': 'yiyue'}
    for file_name in ['春季数据（四月）平均', '夏季数据（七月）平均', '秋季数据（十月）平均', '冬季数据（一月）平均']:
        data_name = season_mapping[file_name]
        data = pd.read_csv(
            open(f'../data/regularization_factor/metrics/{file_name}/{data_name}_average_all.csv',
                 encoding='utf-8'))
        data.set_index(data.columns[0], inplace=True)

        rmse_concat = ['0.1_rmse', '0.2_rmse', '0.3_rmse', '0.4_rmse', '0.5_rmse', '0.6_rmse', '0.7_rmse', '0.8_rmse']
        x = pd.Series(
            ['0.1missing rate', '0.2missing rate', '0.3missing rate', '0.4missing rate', '0.5missing rate',
             '0.6missing rate',
             '0.7missing rate', '0.8missing rate'])
        # l2_0001 = data.loc['gain(0.001l2)', rmse_concat]
        l2_0005 = data.loc['gain(0.005l2)', rmse_concat]
        l2_001 = data.loc['gain(0.01l2)', rmse_concat]
        l2_005 = data.loc['gain(0.05l2)', rmse_concat]
        l2_01 = data.loc['gain(0.1l2)', rmse_concat]
        N = len(x)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        #    0
        #  /   \
        # 3     1    [0,1,2,3,0]才能画出一个首尾相连的多边形，一共N+1个点
        #  \   /
        #    2
        theta = np.concatenate((theta, [theta[0]]), axis=0)  # 默认axis=0
        # l2_0001 = np.concatenate((l2_0001, [l2_0001[0]]))
        l2_0005 = np.concatenate((l2_0005, [l2_0005[0]]))
        l2_001 = np.concatenate((l2_001, [l2_001[0]]))
        l2_005 = np.concatenate((l2_005, [l2_005[0]]))
        l2_01 = np.concatenate((l2_01, [l2_01[0]]))
        x = np.concatenate((x, [x[0]]))
        # 多边形雷达图
        # 隐藏默认圆形框线，手动绘制多边形等间距框线
        plt.figure(figsize=(30, 12))
        rad = plt.subplot(111, polar=True)
        # 画形状相同，顶点间等距的多边形
        # 0，20，40，60，80，100
        if data_name == 'yiyue':
            for i in np.arange(0, 1.75 + 0.25, 0.25):
                rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
            # 用直线连接起这些多边形同一角度的顶点
            for i in range(N):
                rad.plot([theta[i], theta[i]], [0, 1.75],
                         '-.', lw=1, color='black', alpha=0.4)
        elif data_name == 'shiyue':
            for i in np.arange(0, 1.12 + 0.16, 0.16):
                rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
            # 用直线连接起这些多边形同一角度的顶点
            for i in range(N):
                rad.plot([theta[i], theta[i]], [0, 1.12],
                         '-.', lw=1, color='black', alpha=0.4)
        else:
            for i in np.arange(0, 1.4 + 0.2, 0.2):
                rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
            # 用直线连接起这些多边形同一角度的顶点
            for i in range(N):
                rad.plot([theta[i], theta[i]], [0, 1.4],
                         '-.', lw=1, color='black', alpha=0.4)
        # 添加径向刻度标签（靠上方统一显示）
        if data_name == 'yiyue':
            label_r = 1.75
            step = 0.25
        elif data_name == 'shiyue':
            label_r = 1.12
            step = 0.16
        else:
            label_r = 1.4
            step = 0.2
        for r in np.arange(step, label_r + 0.01, step):
            rad.text(np.pi / 2, r - 0.1, f"{r:.1f}", ha='center', va='center', fontsize=25, fontweight='bold')
        # 隐藏黑色圆边界线
        rad.spines['polar'].set_visible(False)
        # 隐藏默认圆形网格线
        rad.grid(False)
        rad.set_yticklabels([])  # 隐藏径向刻度标签

        # rad.plot(theta, l2_0001, color='r', alpha=1, linewidth=2,
        #          label='Penalty parameters of 0.001',
        #          linestyle='-', marker='p', markersize=8)
        rad.plot(theta, l2_0005, color='#018A67', alpha=1, linewidth=2,
                 label='L2 Penalty parameters of 0.005',
                 linestyle='-', marker='p', markersize=8)
        rad.plot(theta, l2_001, color='#1868B2', alpha=1, linewidth=2,
                 label='L2 Penalty parameters of 0.01',
                 linestyle='-', marker='o', markersize=8)
        rad.plot(theta, l2_005, color='#DE582B', alpha=1, linewidth=2,
                 label='L2 Penalty parameters of 0.05',
                 linestyle='-', marker='d', markersize=8)
        rad.plot(theta, l2_01, color='#F3A332', alpha=1, linewidth=2,
                 label='L2 Penalty parameters of 0.1',
                 linestyle='-', marker='*', markersize=10)

        # 标注每个点的值
        # for i, m in zip(theta, l2_0001):
        #     txt = rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='g')
        #     txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='g')])
        # for i, m in zip(theta, l2_0005):
        #     txt = rad.text(i, m - 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='r')
        #     txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='r')])
        # for i, m in zip(theta, l2_001):
        #     txt = rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='b')
        #     txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='b')])
        # for i, m in zip(theta, l2_005):
        #     txt = rad.text(i, m - 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='y')
        #     txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='y')])
        # for i, m in zip(theta, l2_01):
        #     txt = rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=22, color='m')
        #     txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='m')])

        # rad.set_thetagrids(theta * 180 / npy.pi, x)
        rad.set_thetagrids(theta[:-1] * 180 / np.pi, x[:-1])

        # 设置角度刻度标签，避免与坐标轴重叠
        # 通过调整 theta 和 x 的位置，确保标签不与坐标轴重叠
        rad.set_xticks(theta[:-1])  # 设置刻度位置
        rad.set_xticklabels(x[:-1], fontsize=35, fontweight='bold')  # 设置刻度标签
        for label in rad.get_xticklabels():
            label.set_y(label.get_position()[1] - 0.4)  # 负值让标签向外移动

        rad.set_theta_zero_location('S')  # 设置0刻度位置
        if data_name == 'yiyue':
            rad.set_rlim(0, 1.75)  # 设置坐标刻度范围
        elif data_name == 'shiyue':
            rad.set_rlim(0, 1.12)  # 设置坐标刻度范围
        else:
            rad.set_rlim(0, 1.4)
        rad.set_rlabel_position(60)  # 设置坐标显示角度
        # plt.gca().xaxis.label.set_fontweight('bold')
        # plt.gca().yaxis.label.set_fontweight('bold')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, prop={'weight': 'bold', 'size': 25})
        # plt.legend(loc='best', prop={'weight': 'bold', 'size': 16})
        # plt.show()
        # 存储图形
        save_path = f'plots/regularization_factor/{data_name}正则化系数的敏感性分析雷达图.pdf'
        plt.savefig(save_path, dpi=300)
        plt.close()  # 关闭当前图形，防止重叠


# 基于改进步骤的提升百分比图
def plot_improvements():
    plt.rc("font", family='DengXian')  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False  # 负号报错
    season_mapping = {
        '春季数据（四月）平均': 'siyue',
        '夏季数据（七月）平均': 'qiyue',
        '秋季数据（十月）平均': 'shiyue',
        '冬季数据（一月）平均': 'yiyue'}

    for file_name in ['春季数据（四月）平均', '夏季数据（七月）平均', '秋季数据（十月）平均', '冬季数据（一月）平均']:
        # model_improvements = {
        #     'Model A': [0, 0, 0],
        #     'Model B': [0, 0, 0],
        #     'Model C': [0, 0, 0],
        #     'Model D': [0, 0, 0],
        #     'Mine Model': [0, 0, 0]}
        model_improvements = {
            '模型A': [0, 0, 0],
            '模型B': [0, 0, 0],
            '模型C': [0, 0, 0],
            '模型D': [0, 0, 0],
            '模型E': [0, 0, 0]}
        data_name = season_mapping[file_name]
        data = pd.read_csv(open(f'D:/Desktop/我的插补/data/Ablation/metrics/{file_name}/{data_name}_average_all.csv',
                                encoding='utf-8'))
        data.set_index(data.columns[0], inplace=True)
        低缺失 = data.loc[:, 'rmse.1']
        中缺失 = data.loc[:, 'rmse.4']
        高缺失 = data.loc[:, 'rmse.7']
        Baseline_低缺失 = 低缺失['gain(原始)']
        Baseline_中缺失 = 中缺失['gain(原始)']
        Baseline_高缺失 = 高缺失['gain(原始)']
        model_A_低缺失 = 低缺失['gain(+l2)']
        model_A_中缺失 = 中缺失['gain(+l2)']
        model_A_高缺失 = 高缺失['gain(+l2)']
        model_B_低缺失 = 低缺失['gain(+自编码位置编码)']
        model_B_中缺失 = 中缺失['gain(+自编码位置编码)']
        model_B_高缺失 = 高缺失['gain(+自编码位置编码)']
        model_C_低缺失 = 低缺失['gain(+多重)']
        model_C_中缺失 = 中缺失['gain(+多重)']
        model_C_高缺失 = 高缺失['gain(+多重)']
        model_D_低缺失 = 低缺失['gain(+l2+自编码位置编码)']
        model_D_中缺失 = 中缺失['gain(+l2+自编码位置编码)']
        model_D_高缺失 = 高缺失['gain(+l2+自编码位置编码)']
        mine_model_低缺失 = 低缺失['gain(+l2+自编码位置编码+多重)']
        mine_model_中缺失 = 中缺失['gain(+l2+自编码位置编码+多重)']
        mine_model_高缺失 = 高缺失['gain(+l2+自编码位置编码+多重)']
        # 记录提升
        model_A_低缺失_提升 = (model_A_低缺失 - Baseline_低缺失) / Baseline_低缺失
        model_A_中缺失_提升 = (model_A_中缺失 - Baseline_中缺失) / Baseline_中缺失
        model_A_高缺失_提升 = (model_A_高缺失 - Baseline_高缺失) / Baseline_高缺失
        model_B_低缺失_提升 = (model_B_低缺失 - Baseline_低缺失) / Baseline_低缺失
        model_B_中缺失_提升 = (model_B_中缺失 - Baseline_中缺失) / Baseline_中缺失
        model_B_高缺失_提升 = (model_B_高缺失 - Baseline_高缺失) / Baseline_高缺失
        model_C_低缺失_提升 = (model_C_低缺失 - Baseline_低缺失) / Baseline_低缺失
        model_C_中缺失_提升 = (model_C_中缺失 - Baseline_中缺失) / Baseline_中缺失
        model_C_高缺失_提升 = (model_C_高缺失 - Baseline_高缺失) / Baseline_高缺失
        model_D_低缺失_提升 = (model_D_低缺失 - Baseline_低缺失) / Baseline_低缺失
        model_D_中缺失_提升 = (model_D_中缺失 - Baseline_中缺失) / Baseline_中缺失
        model_D_高缺失_提升 = (model_D_高缺失 - Baseline_高缺失) / Baseline_高缺失
        mine_model_低缺失_提升 = (mine_model_低缺失 - Baseline_低缺失) / Baseline_低缺失
        mine_model_中缺失_提升 = (mine_model_中缺失 - Baseline_中缺失) / Baseline_中缺失
        mine_model_高缺失_提升 = (mine_model_高缺失 - Baseline_高缺失) / Baseline_高缺失
        # model_improvements['Model A'][0] = (-model_A_低缺失_提升)
        # model_improvements['Model A'][1] = (-model_A_中缺失_提升)
        # model_improvements['Model A'][2] = (-model_A_高缺失_提升)
        # model_improvements['Model B'][0] = (-model_B_低缺失_提升)
        # model_improvements['Model B'][1] = (-model_B_中缺失_提升)
        # model_improvements['Model B'][2] = (-model_B_高缺失_提升)
        # model_improvements['Model C'][0] = (-model_C_低缺失_提升)
        # model_improvements['Model C'][1] = (-model_C_中缺失_提升)
        # model_improvements['Model C'][2] = (-model_C_高缺失_提升)
        # model_improvements['Model D'][0] = (-model_D_低缺失_提升)
        # model_improvements['Model D'][1] = (-model_D_中缺失_提升)
        # model_improvements['Model D'][2] = (-model_D_高缺失_提升)
        # model_improvements['Mine Model'][0] = (-mine_model_低缺失_提升)
        # model_improvements['Mine Model'][1] = (-mine_model_中缺失_提升)
        # model_improvements['Mine Model'][2] = (-mine_model_高缺失_提升)
        model_improvements['模型A'][0] = (-model_A_低缺失_提升)
        model_improvements['模型A'][1] = (-model_A_中缺失_提升)
        model_improvements['模型A'][2] = (-model_A_高缺失_提升)
        model_improvements['模型B'][0] = (-model_B_低缺失_提升)
        model_improvements['模型B'][1] = (-model_B_中缺失_提升)
        model_improvements['模型B'][2] = (-model_B_高缺失_提升)
        model_improvements['模型C'][0] = (-model_C_低缺失_提升)
        model_improvements['模型C'][1] = (-model_C_中缺失_提升)
        model_improvements['模型C'][2] = (-model_C_高缺失_提升)
        model_improvements['模型D'][0] = (-model_D_低缺失_提升)
        model_improvements['模型D'][1] = (-model_D_中缺失_提升)
        model_improvements['模型D'][2] = (-model_D_高缺失_提升)
        model_improvements['模型E'][0] = (-mine_model_低缺失_提升)
        model_improvements['模型E'][1] = (-mine_model_中缺失_提升)
        model_improvements['模型E'][2] = (-mine_model_高缺失_提升)
        # 绘制水平条形图
        # species = ['Low Missing Rate', 'Medium Missing Rate', 'High Missing Rate']
        species = ['低缺失率', '中缺失率', '高缺失率']
        y = np.arange(len(species))  # 标签位置
        height = 0.18  # 条形高度
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained', figsize=(20, 12))

        for attribute, measurement in model_improvements.items():
            offset = height * multiplier
            rects = ax.barh(y + offset, measurement, height, label=attribute, alpha=0.5)
            ax.bar_label(rects, padding=3, fontsize=30, fmt=lambda x: f"{x * 100:.2f}%", fontweight='bold')
            multiplier += 1

        # 添加标签和标题
        # ax.set_xlabel('Percentage Improvement', fontsize=12, fontweight='bold')
        ax.set_xlabel('性能提升百分比', fontsize=50, fontweight='bold')
        # ax.set_title('Comparison of Model Improvements in Different Missing Rate Levels Based on Improvement Steps',
        #              fontsize=12, fontweight='bold')
        ax.set_yticks(y + height * (len(model_improvements) / 2), species, fontsize=25, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
        if data_name == 'siyue':
            ax.set_xlim(-1.5, 1)
        elif data_name == 'qiyue':
            ax.set_xlim(-2.2, 1)
        elif data_name == 'shiyue':
            ax.set_xlim(-2.3, 1)
        elif data_name == 'yiyue':
            ax.set_xlim(-2.3, 1)
        # 调整图例到左上角并倒序排列
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(0, 1), ncol=1,
                  prop={'weight': 'bold', 'size': 30})
        # y 轴刻度标签显示为百分比 自适应调整刻度
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
        ax.tick_params(axis='both', which='major', labelsize=40, width=2)

        labels = [label.get_text() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontsize=40, fontweight='bold')
        # ax.xaxis.label.set_weight('bold')
        ax.spines['bottom'].set_linewidth(1.5)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)  # 设置上部坐标轴的粗细

        # plt.show()
        # 存储图形
        # save_path = f'plots/Ablation/{data_name}基于改进步骤的对比模型在不同缺失程度下的rmse的提升百分比柱状图.pdf'
        save_path = f'plots/Ablation_cn/{data_name}基于改进步骤的对比模型在不同缺失程度下的rmse的提升百分比柱状图.pdf'
        plt.savefig(save_path)
        plt.close()  # 关闭当前图形，防止重叠


def Friedman_Test():
    method_names = ['Proposed Model', 'EM', 'KNN', 'LOCF', 'Mean', 'Median', 'MICE', 'MissForest']
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        all_mae = []  # shape: (num_tasks, num_methods)

        # 遍历任务
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for random_seed in [1, 2, 3]:
                # 每个方法一个mae（平均或最后一个）
                maes = []
                for method in ['gain(+l2+自编码位置编码+多重)', 'em', 'knn', 'locf', 'mean', 'median', 'mice',
                               'missforest']:
                    file_path = f'../data/metrics/{missing_rate}missing_rate/{data_name}/{method}_{random_seed}.csv'
                    mae = pd.read_csv(file_path)['mae'].mean()  # 使用均值或指定位置
                    maes.append(mae)
                all_mae.append(maes)

        all_mae = np.array(all_mae)  # shape: (任务数量, 方法数量)

        # Friedman 检验
        stat, p = friedmanchisquare(*[all_mae[:, i] for i in range(all_mae.shape[1])])
        print(f"[{data_name}] Friedman 检验结果: χ² = {stat:.3f}, p = {p:.4f}")

        # 计算每个任务下的排名
        ranks = np.argsort(np.argsort(all_mae, axis=1), axis=1) + 1  # shape: (num_tasks, num_methods)

        # 为箱线图准备数据：每列是一个方法在所有任务下的排名
        data_for_boxplot = [ranks[:, i] for i in range(ranks.shape[1])]

        # 绘制箱线图
        plt.figure(figsize=(16, 9))
        sns.boxplot(data=data_for_boxplot, showfliers=False, linewidth=1.5)

        ax = plt.gca()  # 获取当前坐标轴
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_linewidth(1.5)
        plt.tick_params(axis='both', which='major', labelsize=12, width=2)
        plt.xticks(ticks=np.arange(len(method_names)), labels=method_names,
                   rotation=45, ha='right', fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        plt.ylabel('Rank', fontsize=25, fontweight='bold')
        plt.tight_layout()
        # plt.show()
        # 存储图形
        save_path = f'plots/Friedman_Test/{data_name}_friedman_test.pdf'
        # plt.savefig(save_path, dpi=300)


def plot_error_bar_chart():
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.2, 0.5, 0.8]:
            for random_seed in [1]:
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                imputed_data = pd.read_csv(
                    f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                    header=None)
                mask = data_m == 0
                x = ori_data_x[mask]
                y = imputed_data.values[mask]
                # 计算误差
                error = x - y
                color_map = {
                    'yiyue': '#9DBAD2',
                    'siyue': '#9C6BA3',
                    'qiyue': '#98C897',
                    'shiyue': '#FE9C9D'}
                miss_map = {
                    0.2: '20%',
                    0.5: '50%',
                    0.8: '80%'}
                # 设置全局字体为 Times New Roman
                # plt.rcParams['font.family'] = 'Times New Roman'
                plt.rc("font", family='DengXian')  # 解决中文乱码
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 调整刻度字体大小、加粗刻度线和刻度字体
                # 绘制误差的直方图
                plt.figure(figsize=(12, 10))
                sns.histplot(error, bins=30, kde=False, alpha=1, color=color_map[data_name], stat='density',
                             label=r'$\mathbf{Error}$' '\n' r'$\mathbf{Distribution}$')
                # 拟合正态分布并绘制曲线
                mu, std = norm.fit(error)  # 估算误差的均值和标准差
                xmin, xmax = plt.xlim()  # 获取当前x轴的范围
                n = np.linspace(xmin, xmax, 100)
                p = norm.pdf(n, mu, std)  # 计算正态分布的概率密度
                plt.plot(n, p, linewidth=2, color='red', alpha=0.8,
                         label=(r'$\mathbf{Normal}$' '\n'
                                r'$\mathbf{Distribution}$' '\n'
                                r'$\boldsymbol{\mu=' + f'{mu:.2f}' + r'},$' '\n'
                                                                     r'$\boldsymbol{\sigma=' + f'{std:.2f}' + r'}.$'))
                # 添加标签和标题
                # plt.title('Error Distribution with Normal Fit', fontsize=15)
                plt.xlabel(f'{miss_map[missing_rate]}Missingness Error', fontsize=40, labelpad=9, fontweight='bold')
                plt.ylabel('Density', fontsize=40, labelpad=10, fontweight='bold')
                # plt.minorticks_on()  # 添加次刻度线
                # 获取当前图形的坐标轴对象
                ax = plt.gca()
                # 设置坐标轴边框线条加粗
                ax.spines['top'].set_linewidth(2)  # 设置上边框加粗
                ax.spines['right'].set_linewidth(2)  # 设置右边框加粗
                ax.spines['bottom'].set_linewidth(2)  # 设置下边框加粗
                ax.spines['left'].set_linewidth(2)  # 设置左边框加粗
                # 调整刻度字体大小、加粗刻度线和刻度字体
                ax.tick_params(axis='both', which='major', labelsize=16, width=2)  # 主刻度字体加粗，线条加粗
                # ax.tick_params(axis='both', which='minor', labelsize=14, width=1)  # 次刻度字体加粗，线条加粗
                # 设置刻度字体加粗
                ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=30)
                ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=30)
                plt.legend(prop={'weight': 'bold', 'size': 25}, frameon=False, loc='upper right',
                           bbox_to_anchor=(1.0, 1.0),  # 精确定位右上角
                           borderaxespad=0,  # 控制图例和坐标轴边的间距（越小越贴边）
                           handletextpad=0.5,  # 图例图形与文字之间的水平间距（默认是 0.8）
                           handlelength=1.5,  # 控制图例中线条/图形的长度（默认是 2）
                           handleheight=0.5)  # 控制图例中方块的高度（对柱状图效果明显）)
                # 显示图形
                # plt.show()
                # 存储图形
                save_path = f'plots/Error_Distribution/{data_name}/{data_name}的{missing_rate}缺失率的误差分布图_{random_seed}.pdf'
                plt.savefig(save_path)
                plt.close()  # 关闭当前图形，防止重叠
                print(
                    f'\033[31m{data_name}的{missing_rate}缺失率的误差分布图_{random_seed}已保存\033[0m')


# 讨论中，不同缺失机制下模型的性能柱状图
def dif_mechanism_plot_discuss():
    for missing_rate in [0.2, 0.4]:
        # 假设三组文件是以下路径（可按需修改）
        file_mcar = f'../data/Discussion/Different_missing_mechanisms/metrics/siyue_{missing_rate}_mcar_gain(+l2+自编码位置编码+多重)_1.csv'
        file_mar = f'../data/Discussion/Different_missing_mechanisms/metrics/siyue_{missing_rate}_mar_gain(+l2+自编码位置编码+多重)_1.csv'
        file_mnar = f'../data/Discussion/Different_missing_mechanisms/metrics/siyue_{missing_rate}_mnar_gain(+l2+自编码位置编码+多重)_1.csv'

        # 读取每个文件，并添加类型标签
        df_mcar = pd.read_csv(file_mcar)
        df_mcar['type'] = 'MCAR'

        df_mar = pd.read_csv(file_mar)
        df_mar['type'] = 'MAR'

        df_mnar = pd.read_csv(file_mnar)
        df_mnar['type'] = 'MNAR'

        df_all = pd.concat([df_mcar, df_mar, df_mnar], ignore_index=True)

        # 准备绘图数据
        metrics = ['rmse', 'mae', 'mape']
        x = np.arange(len(df_all))  # 三个缺失机制
        colors = ['#2C91E0', '#3ABF99', '#F0A73A']
        width = 0.2

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 9))
        rects_list = []
        for i, metric in enumerate(metrics):
            rects = ax.bar(x + i * width, df_all[metric], width, label=metric.upper(), color=colors[i], alpha=1,
                           edgecolor='black', linewidth=1.5)
            rects_list.append(rects)
        # 添加数值标签
        for rects in rects_list:
            ax.bar_label(rects,
                         padding=2,
                         fontsize=18,
                         fmt="%.3f",
                         fontweight='bold')

        # 坐标轴标签设置
        ax.spines['bottom'].set_linewidth(1.5)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)  # 设置上部坐标轴的粗细
        plt.tick_params(axis='both', which='major', width=2)  # 主刻度字体加粗，线条加粗
        ax.set_xticklabels(df_all['type'], fontsize=25, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=25, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xlabel('Missing Mechanism', fontsize=30, fontweight='bold')
        ax.set_ylabel('Value', fontsize=30, fontweight='bold')
        ax.legend(loc='upper left', prop={'size': 20, 'weight': 'bold'})

        plt.tight_layout()
        # plt.show()
        # 存储图形
        save_path = f'../data/plots/Discussion/Different_missing_mechanisms/不同缺失机制下_{missing_rate}的模型性能柱状图_discuss.pdf'
        plt.savefig(save_path, dpi=300)


# 讨论中，对比时间序列模型性能对比图
def time_series_plot_discuss():
    for data_name in ['yiyue_1800', 'siyue_1800', 'qiyue_1800', 'shiyue_1800']:
        df = pd.read_csv(f'Discussion/ts_compare/metrics/{data_name}_all.csv',
                         index_col=0)
        rmse = df[['rmse_0.1', 'rmse_0.2', 'rmse_0.3', 'rmse_0.4']]
        rmse.columns = ['10%', '20%', '30%', '40%']
        brits = rmse.loc['BRITS', :]
        saits = rmse.loc['SAITS', :]
        usgan = rmse.loc['USGAN', :]
        proposed = rmse.loc['proposed', :]
        # 创建图形
        plt.figure(figsize=(20, 12))  # 设置图形大小
        # 设置颜色
        colors = ['#1999B2', '#FEE066', '#95BCE5', '#AD0B08']
        # 设置线条样式
        linestyles = ['-', '-', '-', '-']
        # 设置线条宽度
        linewidths = [4, 4, 4, 4]
        # 设置标记
        markers = ['o', 's', 'p', '*']
        # 设置 x 轴刻度位置
        xticks = [0.1, 0.2, 0.3, 0.4]
        # 设置 x 轴刻度标签
        xticklabels = ['10%', '20%', '30%', '40%']
        # 绘制图形
        plt.plot(xticks, brits, label='BRITS', color=colors[0], linestyle=linestyles[0], linewidth=linewidths[0],
                 marker=markers[0], markersize=10)  # BRITS
        plt.plot(xticks, saits, label='SAITS', color=colors[1], linestyle=linestyles[1], linewidth=linewidths[1],
                 marker=markers[1], markersize=10)  # SAITS
        plt.plot(xticks, usgan, label='US-GAN', color=colors[2], linestyle=linestyles[2], linewidth=linewidths[2],
                 marker=markers[2], markersize=10)  # USGAN
        plt.plot(xticks, proposed, label='proposed model', color=colors[3], linestyle=linestyles[3],
                 linewidth=linewidths[3], marker=markers[3], markersize=15)  # Proposed
        # 设置坐标轴边框线条加粗
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
        plt.tick_params(axis='both', which='major', width=2)  # 主刻度字体加粗，线条加粗
        # 设置 x 轴刻度位置
        plt.xticks(xticks, xticklabels, fontsize=35, fontweight='bold')
        # 设置 y 轴刻度位置
        plt.yticks(fontsize=35, fontweight='bold')
        # 强制设置 y 轴刻度数量为 5
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        # 设置 x 轴标签
        plt.xlabel('Misssing rate', fontsize=45, fontweight='bold', labelpad=-2)
        # 设置 y 轴标签
        plt.ylabel('RMSE', fontsize=45, fontweight='bold', labelpad=3)
        # 设置图例，横着摆放在标题下方、坐标轴外
        plt.legend(
            loc='upper center',  # 锚点位置
            bbox_to_anchor=(0.5, 1.15),  # 将图例放置在标题下方
            ncol=5,  # 横着摆放，5 列
            prop={'size': 30, 'weight': 'bold'},  # 字体大小
        )

        # 调整图形布局，为图例留出空间
        plt.subplots_adjust(top=0.8)  # 调整底部留出空间
        # 添加网格线
        # plt.grid(True, linestyle='--', alpha=0.6)
        # 显示图形
        # plt.show()
        # 存储图形
        save_path = f'plots/Discussion/ts_compare/{data_name}_ts_compare_rmse.eps'
        plt.savefig(save_path, dpi=300)


# 讨论中，aqi数据的插补曲线图

def plot_imputation_aqi():
    plt.rc("font", family='DengXian')  # 解决中文乱码
    for column in [0, 1, 2, 3, 4]:
        ori_data_x, missing_data, data_m = data_loader('Beijing_aqi', 0.4, 1)
        imputation_data = np.loadtxt(
            f'Discussion\\aqi\\imputation_data\\Beijing_aqi_0.4_gain(+l2+自编码位置编码+多重)_aqi_1.csv',
            delimiter=',')
        imputation_data_1 = imputation_data[:, column]
        ori_data_x_1 = ori_data_x[:, column]
        missing_data_1 = missing_data[:, column]
        data_m_1 = data_m[:, column]
        mask = data_m_1 == 1
        time = np.arange(0, len(ori_data_x_1), 1)

        # 绘制折线图
        plt.figure(figsize=(20, 12))
        m, n = 1, 1
        for i in range(len(ori_data_x_1) - 1):
            if mask[i] and mask[i + 1]:
                plt.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='#377EB9', marker='',
                         zorder=10)
            elif not mask[i] and not mask[i + 1]:
                plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                         markersize=2, label='Missing Data' if m == 1 else "")  # color='#198E5C'
                plt.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                         label='Imputed Data' if m == 1 else "", zorder=10)  # color='#EC050E'
                # plt.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                #          markersize=2, label='缺失数据' if m == 1 else "")  # color='#198E5C'
                # plt.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                #          label='插补数据' if m == 1 else "", zorder=10)  # color='#EC050E'
                m = 0
            else:
                # plt.plot([time[i], time[i + 1]],
                #          [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                #           ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                #          color='#377EB9', label='原始数据' if n == 1 else "", zorder=10)  # color='#5C8DC7'
                plt.plot([time[i], time[i + 1]],
                         [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                          ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]], linestyle='-',
                         color='#377EB9', label='Original Data' if n == 1 else "",
                         zorder=10)  # color='#5C8DC7'
                n = 0
        # 获取当前图形的坐标轴对象
        ax = plt.gca()
        # 设置坐标轴边框线条加粗
        ax.spines['top'].set_linewidth(1.5)  # 设置上边框加粗
        ax.spines['right'].set_linewidth(1.5)  # 设置右边框加粗
        ax.spines['bottom'].set_linewidth(1.5)  # 设置下边框加粗
        ax.spines['left'].set_linewidth(1.5)  # 设置左边框加粗
        ax.tick_params(axis='both', which='major', labelsize=40, width=2)  # 主刻度字体加粗，线条加粗
        # 设置刻度字体加粗
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=40)
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=40)
        plt.legend(prop={'weight': 'bold', 'size': 30}, loc='upper left', bbox_to_anchor=(0.0001, 0.7))
        # plt.title('Line Plot with Missing Data and Interpolation')
        plt.xlabel('Time(1 hour)', fontsize=50, fontweight='bold')
        # plt.xlabel('时间点（十分钟间隔）', fontsize=50, fontweight='bold')
        plt.ylabel('AQI', fontsize=50, fontweight='bold')
        # plt.ylabel('风速(m/s)', fontsize=50, fontweight='bold')

        # 添加放大镜效果
        # 选择要放大的区域（这里假设放大中间1/3的区域）
        x1, x2 = len(ori_data_x_1) // 10, 2 * len(ori_data_x_1) // 10
        y1 = min(ori_data_x_1[x1:x2]) - 0.5
        y2 = max(ori_data_x_1[x1:x2]) + 0.5
        # 创建放大镜子图
        axins = ax.inset_axes((0.005, 0.69, 0.3, 0.3))  # 放大镜位置和大小
        # **加粗放大镜的边框**
        for spine in axins.spines.values():
            spine.set_linewidth(1)  # 可以调整数值来改变粗细
        for i in range(len(ori_data_x_1) - 1):
            if mask[i] and mask[i + 1] and x1 <= i <= x2:
                axins.plot(time[i:i + 2], missing_data_1[i:i + 2], linestyle='-', color='#377EB9',
                           marker='', zorder=10)
            elif not mask[i] and not mask[i + 1] and x1 <= i <= x2:
                axins.plot(time[i:i + 2], ori_data_x_1[i:i + 2], linestyle='--', color='#198E5C', marker='',
                           markersize=2)
                axins.plot(time[i:i + 2], imputation_data_1[i:i + 2], linestyle='-', color='#EC050E',
                           zorder=10)
            elif x1 <= i <= x2:
                axins.plot([time[i], time[i + 1]],
                           [missing_data_1[i] if mask[i] else ori_data_x_1[i],
                            ori_data_x_1[i + 1] if not mask[i + 1] else missing_data_1[i + 1]],
                           linestyle='-', color='#377EB9', zorder=10)

        # 设置放大镜的坐标范围
        axins.set_xlim(time[x1], time[x2 - 1])
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.grid(True, linestyle='--', alpha=0.7)

        # 在主图中标记放大区域
        # ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, linewidth=1)

        # 在主图中添加放大区域的矩形标记
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='black', facecolor='none',
                             linestyle='--')
        ax.add_patch(rect)
        # 连接放大镜区域与主图
        # rect = plt.Rectangle((time[x1], y1), time[x2 - 1] - time[x1], y2 - y1, linewidth=1.5,
        #                      edgecolor='black', facecolor='none', linestyle='--')
        # ax.add_patch(rect)
        plt.tight_layout()
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2.0)
        # plt.show()
        # 存储图形
        save_path = f'plots/Discussion/aqi/第{column + 1}个站点的插补对比图.eps'
        plt.savefig(save_path, dpi=300)
        plt.close()  # 关闭当前图形，防止重叠


def plot_var_prediction_comparison(original_data, imputed_data, steps=30):
    # 设置列名：风机编号
    columns = [f"WT{i + 1}" for i in range(original_data.shape[1])]

    # 构建 DataFrame
    ori_df = pd.DataFrame(original_data, columns=columns)
    imp_df = pd.DataFrame(imputed_data, columns=columns)

    # 拟合 VAR 模型
    ori_model = VAR(ori_df).fit(maxlags=15, ic='aic')
    imp_model = VAR(imp_df).fit(maxlags=15, ic='aic')

    # 模型预测（使用最后 k_ar 步预测未来 steps 步）
    ori_forecast = ori_model.forecast(ori_df.values[-ori_model.k_ar:], steps=steps)
    imp_forecast = imp_model.forecast(imp_df.values[-imp_model.k_ar:], steps=steps)

    # 真实值（原始数据后 steps 步）
    ori_true = ori_df.values[-steps:]
    imp_true = imp_df.values[-steps:]

    # 可视化每个风机
    for i in range(original_data.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(ori_true[:, i], '--', label="Original True", color='black')
        plt.plot(ori_forecast[:, i], label="Original VAR Forecast", color='blue')
        plt.plot(imp_true[:, i], '--', label="Imputed True", color='gray')
        plt.plot(imp_forecast[:, i], label="Imputed VAR Forecast", color='green')
        plt.title(f"VAR Forecast Comparison - Turbine {i + 1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Wind Speed")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 插补前后的ACF和PACF对比
def plot_acf_pacf_comparison():
    color_map = {
        0.2: '#5BB5AC',
        0.5: '#D8B365',
        0.8: '#DE526C'}
    for data_name in ['siyue']:
        for missing_rate in [0.2, 0.5, 0.8]:
            for idx in [0, 1, 2, 3, 4]:
                color = color_map[missing_rate]
                random_seed = 1
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                imputed_data = pd.read_csv(
                    f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                    header=None)
                ori_series = ori_data_x[:, idx]
                imp_series = imputed_data.values[:, idx]
                fig, axs = plt.subplots(2, 2, figsize=(16, 9))
                for ax in axs.flatten():
                    # 设置坐标轴边框线条加粗
                    ax.spines['top'].set_linewidth(2)  # 设置上边框加粗
                    ax.spines['right'].set_linewidth(2)  # 设置右边框加粗
                    ax.spines['bottom'].set_linewidth(2)  # 设置下边框加粗
                    ax.spines['left'].set_linewidth(2)  # 设置左边框加粗
                    ax.tick_params(axis='both', which='major', labelsize=15, width=2)  # 设置刻度大小和宽度
                    # 加粗刻度标签（x轴和y轴）
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontweight('bold')  # 直接设置字体粗细

                lags = 40
                # ACF
                plot_acf(ori_series, lags=lags, ax=axs[0, 0])
                axs[0, 0].set_title('Original ACF', fontsize=20, fontweight='bold')
                axs[0, 0].set_ylabel('Autocorrelation Coefficient', fontsize=15, fontweight='bold')
                # 修改竖线和点
                for line in axs[0, 0].lines:
                    line.set_color(color)
                # 修改中间的柱状填充（Stem）
                for coll in axs[0, 0].collections:
                    coll.set_color(color)
                # 修改置信区间（置信区间是一个 fill_between）
                for patch in axs[0, 0].patches:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.2)  # 可选：调整透明度
                axs[0, 0].collections[0].set_color(color)

                plot_acf(imp_series, lags=lags, ax=axs[0, 1], color=color)
                axs[0, 1].set_title('Imputed ACF', fontsize=20, fontweight='bold')
                # 修改竖线和点
                for line in axs[0, 1].lines:
                    line.set_color(color)
                # 修改中间的柱状填充（Stem）
                for coll in axs[0, 1].collections:
                    coll.set_color(color)
                # 修改置信区间（置信区间是一个 fill_between）
                for patch in axs[0, 1].patches:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.2)  # 可选：调整透明度
                axs[0, 1].collections[0].set_color(color)

                # PACF
                plot_pacf(ori_series, lags=lags, ax=axs[1, 0], method='ywm', color=color)
                axs[1, 0].set_title('Original PACF', fontsize=20, fontweight='bold')
                axs[1, 0].set_ylabel('Partial Autocorrelation Coefficient', fontsize=15, fontweight='bold')
                axs[1, 0].set_xlabel('Lags', fontsize=20, fontweight='bold')
                # 修改竖线和点
                for line in axs[1, 0].lines:
                    line.set_color(color)
                # 修改中间的柱状填充（Stem）
                for coll in axs[1, 0].collections:
                    coll.set_color(color)
                # 修改置信区间（置信区间是一个 fill_between）
                for patch in axs[1, 0].patches:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.2)  # 可选：调整透明度
                axs[1, 0].collections[0].set_color(color)

                plot_pacf(imp_series, lags=lags, ax=axs[1, 1], method='ywm', color=color)
                axs[1, 1].set_title('Imputed PACF', fontsize=20, fontweight='bold')
                axs[1, 1].set_xlabel('Lags', fontsize=20, fontweight='bold')
                # 修改竖线和点
                for line in axs[1, 1].lines:
                    line.set_color(color)
                # 修改中间的柱状填充（Stem）
                for coll in axs[1, 1].collections:
                    coll.set_color(color)
                # 修改置信区间（置信区间是一个 fill_between）
                for patch in axs[1, 1].patches:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.2)  # 可选：调整透明度
                axs[1, 1].collections[0].set_color(color)

                # 隐藏右侧子图的 y 轴刻度和刻度标签
                axs[0, 1].set_yticklabels([])
                axs[1, 1].set_yticklabels([])
                # fig.suptitle(f'Time Dependency Comparison', fontsize=16)
                plt.tight_layout()
                # plt.show()
                # 保存图片
                save_path = f'plots/ACF&PACF/{data_name}/{missing_rate}missing_rate/{data_name}_{missing_rate}_第{idx + 1}风机_ACF_PACF.pdf'
                plt.savefig(save_path, dpi=300)


# K-S检验
def ks_test():
    # 用于存储每个季节的p值
    season_p_values = {
        'yiyue': pd.DataFrame(columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], index=[1, 2, 3, 4, 5]),
        'siyue': pd.DataFrame(columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], index=[1, 2, 3, 4, 5]),
        'qiyue': pd.DataFrame(columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], index=[1, 2, 3, 4, 5]),
        'shiyue': pd.DataFrame(columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], index=[1, 2, 3, 4, 5])
    }
    for data_name in ['yiyue', 'siyue', 'qiyue', 'shiyue']:
        for missing_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for idx in range(5):
                random_seed = 1
                ori_data_x, missing_data, data_m = data_loader(data_name, missing_rate, random_seed)
                imputed_data = pd.read_csv(
                    f'imputation_data\\{missing_rate}missing_rate\\{data_name}\\gain(+l2+自编码位置编码+多重)_{random_seed}.csv',
                    header=None)
                ori_series = ori_data_x[:, idx]
                imp_series = imputed_data.values[:, idx]

                # K-S双样本检验
                ks_stat, p_value = ks_2samp(ori_series, imp_series)
                print(
                    f"[{data_name} - 缺失率{missing_rate} - 风机{idx + 1}] K-S检验结果：statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
                if p_value > 0.05:
                    print("分布差异不显著 ✅")
                else:
                    print("分布差异显著 ❌")

                # 将 p 值存入对应季节和风机的 DataFrame
                season_p_values[data_name].at[idx + 1, missing_rate] = p_value
    for season, df in season_p_values.items():
        file_name = f"KS_test/{season}_ks_p_values.csv"
        df.to_csv(file_name)
        print(f"导出 {season} 的 K-S 检验 p 值数据帧到 {file_name}")


if __name__ == '__main__':
    plot_imputation_aqi()
