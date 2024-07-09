import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
ax1=plt.subplot(3,2,1)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [5.3057 , 5.1119, 5.0253, 5.6776, 5.6159,6.0481,5.9367,6.4739,6.1823,5.5815,5.6874]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='edl_mae')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mae')

ax1=plt.subplot(3,2,2)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [25.9136,29.5002, 34.7996, 34.7483, 43.2765,48.7587,52.3752,56.3615,52.9374,57.9787,57.8055]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='edl_mis')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mis')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
ax1=plt.subplot(3,2,3)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [7.2441 , 5.5672, 4.8213, 4.7444, 4.7152,4.6529,5.2405,5.0166,5.6501,4.9661,4.7954]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='MT-edl_mae')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mae')

ax1=plt.subplot(3,2,4)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [21.5689,28.9025, 34.8904, 36.0160, 34.9469,62.0227,46.8646,48.7901,50.0873,54.0427,53.4635]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='MT-edl_mis')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mis')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
ax1=plt.subplot(3,2,5)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [7.6937  , 7.0170, 6.3852, 5.5985, 4.9852,4.6391,6.2809,5.3280,5.6858,4.8610,5.0573]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='Edl_dcrnn_0.1_mae')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mae')

ax1=plt.subplot(3,2,6)
x_axis_data = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y_axis_data = [14.8867,18.6510, 23.7204, 28.4082, 28.6073,31.6892,41.2296,40.5278,40.2827,42.5261,44.9818]
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, '--', color='#4169E1', alpha=0.8, linewidth=1, label='Edl_dcrnn_0.1_mis')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('mis')

plt.show()