import pandas as pd
import numpy as npy
import matplotlib.pyplot as plt

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
        open(f'D:/Desktop/我的插补/data/Ablation/metrics/{file_name}/{data_name}_average_all.csv', encoding='utf-8'))
    data.set_index(data.columns[0], inplace=True)

    rmse_concat = ['rmse', 'rmse.1', 'rmse.2', 'rmse.3', 'rmse.4', 'rmse.5', 'rmse.6', 'rmse.7']
    x = pd.Series(
        ['0.1missing rate', '0.2missing rate', '0.3missing rate', '0.4missing rate', '0.5missing rate', '0.6missing rate',
         '0.7missing rate', '0.8missing rate'])
    l1 = data.loc['gain(+l1+自编码位置编码)', rmse_concat]
    l2 = data.loc['gain(+l2+自编码位置编码)', rmse_concat]
    droupout = data.loc['gain(+droupout+自编码位置编码)', rmse_concat]
    N = len(x)
    theta = npy.linspace(0, 2 * npy.pi, N, endpoint=False)
    #    0
    #  /   \
    # 3     1    [0,1,2,3,0]才能画出一个首尾相连的多边形，一共N+1个点
    #  \   /
    #    2
    theta = npy.concatenate((theta, [theta[0]]), axis=0)  # 默认axis=0
    l1 = npy.concatenate((l1, [l1[0]]))
    l2 = npy.concatenate((l2, [l2[0]]))
    droupout = npy.concatenate((droupout, [droupout[0]]))
    x = npy.concatenate((x, [x[0]]))

    # 多边形雷达图
    # 隐藏默认圆形框线，手动绘制多边形等间距框线
    plt.figure(figsize=(10, 10), dpi=120)
    rad = plt.subplot(111, polar=True)
    # 画形状相同，顶点间等距的多边形
    # 0，20，40，60，80，100
    if data_name == 'yiyue':
        for i in npy.arange(0, 2.5 + 0.5, 0.5):
            rad.plot(theta, 9 * [i], '-.', lw=1, color='black', alpha=0.4)
        # 用直线连接起这些多边形同一角度的顶点
        for i in range(N):
            rad.plot([theta[i], theta[i]], [0, 2.5],
                     '-.', lw=1, color='black', alpha=0.4)
    else:
        for i in npy.arange(0, 2 + 0.4, 0.4):
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

    rad.fill(theta, l2, color='g', alpha=0.25)

    rad.plot(theta, l1, color='r', alpha=0.5, linewidth=1.5, label='l1', linestyle='-', marker='o', markersize=3)
    rad.plot(theta, l2, color='g', alpha=0.5, linewidth=1.5, label='l2', linestyle='-', marker='o', markersize=3)
    rad.plot(theta, droupout, color='b', alpha=0.5, linewidth=1.5, label='droupout', linestyle='-', marker='o', markersize=3)
    # 标注每个点的值
    for i, m in zip(theta, l1):
        rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=12, color='r')
    for i, m in zip(theta, l2):
        rad.text(i, m - 0.15, f'{m:.2f}', ha='center', va='center', fontsize=12, color='g')
    for i, m in zip(theta, droupout):
        rad.text(i, m + 0.15, f'{m:.2f}', ha='center', va='center', fontsize=12, color='b')
    # rad.set_thetagrids(theta * 180 / npy.pi, x)
    rad.set_thetagrids(theta[:-1] * 180 / npy.pi, x[:-1])

    # 设置角度刻度标签，避免与坐标轴重叠
    # 通过调整 theta 和 x 的位置，确保标签不与坐标轴重叠
    rad.set_xticks(theta[:-1])  # 设置刻度位置
    rad.set_xticklabels(x[:-1], fontsize=13)  # 设置刻度标签
    # 设置标签的偏移量
    # for label, angle in zip(rad.get_xticklabels(), theta[:-1]):
    #     label.set_horizontalalignment('center')
    #     label.set_verticalalignment('bottom')
    #     label.set_rotation(60)  # 根据需要调整角度

    rad.set_theta_zero_location('S')  # 设置0刻度位置
    if data_name == 'yiyue':
        rad.set_rlim(0, 2.5)  # 设置坐标刻度范围
    else:
        rad.set_rlim(0, 2)
    rad.set_rlabel_position(60)  # 设置坐标显示角度
    rad.set_title("rmse", pad=60)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
    plt.show()
