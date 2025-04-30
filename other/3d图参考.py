# 获取x、y、z数据
# data_name = 'yiyue'
# data_x = np.loadtxt(f'months/{data_name}.csv', delimiter=',')
# data_x = pd.DataFrame(data_x)
# x = np.arange(data_x.shape[1])
# y = np.arange(data_x.shape[0])
# X, Y = np.meshgrid(x, y)
# Z = data_x.values

# 绘制三维散点图
# fig = plt.figure(figsize=(16, 8))
# fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.1)
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(data_x.columns)):
# ax.plot3D(y, Z[:, i], zs=i, zdir='y')
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))  # xyz轴的长宽高
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_major_locator(ticker.MultipleLocator(40))
# ax.invert_xaxis()
# ax.view_init(azim=45)  # elev=20将观察者从z轴正方向向下看20度，azim然后逆时针旋转30度
# ax.xaxis.labelpad = 20  # 标签间距
# ax.set_zlabel('Value')
# ax.set_xlabel('Sampling point')
# ax.set_yticks(x)
# ax.set_yticklabels(data_x.columns)
# fig.align_labels()
# plt.savefig(draw_path + 'v_fdc.tiff', dpi=600, bbox_inches='tight', transparent=True)
# plt.savefig(draw_path + 'v_fdc.svg', dpi=600, bbox_inches='tight', transparent=True)
# plt.show()