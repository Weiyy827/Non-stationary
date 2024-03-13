import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义阵元参数
N = 8  # 阵元数量
d = 0.5  # 阵元间距（以波长为单位）
frequency = 3e8  # 频率（Hz）
wavelength = 3e8 / frequency  # 波长（m）

# 计算方向因子
def array_factor(theta, phi):
    k = 2 * np.pi / wavelength  # 波数
    psi = k * d * np.sin(theta) * np.cos(phi)  # 相位差
    af = np.abs(np.sum(np.exp(1j * np.arange(N) * psi)))  # 方向因子
    return af

# 创建网格
theta, phi = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100))

# 计算方向图
af = np.vectorize(array_factor)(theta, phi)

# 转换为直角坐标
x = af * np.sin(theta) * np.cos(phi)
y = af * np.sin(theta) * np.sin(phi)
z = af * np.cos(theta)

# 绘制三维方向图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
