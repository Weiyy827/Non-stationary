import commpy
import numpy as np
from scipy import constants
from scipy.io import savemat

import channel_model
import scenario
from config import fc, bw, n

# 参数设置

# 生成发送信号
bits = np.random.binomial(1, 0.5, n)
qpsk = commpy.PSKModem(4)
x = qpsk.modulate(bits)
# 上采样到采样率

# 生成卫星
Sat = scenario.Satellite(height=500e3, azimuth=30, elevation=30)

# 经纬度和半径决定本地GCS的原点，latitude纬度，longitude经度
Origin = scenario.Origin(latitude=45, longitude=45)

# Rx天线设置
Rx = scenario.Antenna(
    [10, 0, 1.5],
    [45, 90],
    [0.5, 0, 0],
    45,
    Ant_type="ULA",
    Num=32,
    Delta=constants.c / fc / 2
)
#  地球GCS到本地GCS转换
Sat_GCS_coordinate = Origin.rotation @ (Sat.global_GCS_coordinate - Origin.global_GCS_coordinate)

Tx = scenario.Antenna(
    Sat_GCS_coordinate,
    [0, -90],
    Sat.velocity @ np.transpose(Origin.rotation),
    45,
    Ant_type="ULA",
    Num=32,
    Delta=constants.c / fc / 2,
)

# 过信道
t = 0
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw, t)

# 存储信号文件
mat_file = 'signal.mat'
savemat(mat_file, {'x': x, 'y': y})
