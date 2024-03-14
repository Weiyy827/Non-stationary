import commpy
import numpy as np

import antenna
import channel_model
from config import fc, bw, n, c

# 参数设置

# 生成发送信号
bits = np.random.binomial(1, 0.5, n)
qpsk = commpy.PSKModem(4)
x = qpsk.modulate(bits)

# 生成天线
Sat = antenna.Satellite(height=500e3, azimuth=0, elevation=np.pi / 6)

# 经纬度和半径决定原点GCS的原点
Origin = antenna.Origin(latitude=39.9062, longitude=116.3912)

# Rx天线设置
Rx = antenna.Antenna([10, 0, 1.5], [0, np.pi / 2], [0.5, 0, 0], np.pi / 4, Ant_type='ULA', Num=32,
                     Delta=c / fc / 2)
#  地球GCS到原点GCS转换
Sat_GCS_coordinate = (Sat.Global_GCS_coordinate - Origin.Global_GCS_coordinate)*Origin.Rotation

Tx = antenna.Antenna(Sat_GCS_coordinate, [np.pi / 2, -np.pi / 2],
                     [Sat.vsat * np.cos(Sat.azimuth), Sat.vsat * np.sin(Sat.azimuth), 0],
                     np.pi / 4, Ant_type='ULA', Num=32, Delta=c / fc / 2)

# 过信道
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw)
