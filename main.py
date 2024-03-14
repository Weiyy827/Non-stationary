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

# 生成卫星
Sat = antenna.Satellite(height=500e3, azimuth=0, elevation=0)

# 经纬度和半径决定本地GCS的原点，latitude纬度，longitude经度
Origin = antenna.Origin(latitude=0, longitude=0)

# Rx天线设置
Rx = antenna.Antenna([10, 0, 1.5], [0, np.pi / 2], [0.5, 0, 0], np.pi / 4, Ant_type='ULA', Num=32,
                     Delta=c / fc / 2)
#  地球GCS到本地GCS转换
Sat_GCS_coordinate = Origin.Rotation @ (Sat.Global_GCS_coordinate - Origin.Global_GCS_coordinate)

Tx = antenna.Antenna(Sat_GCS_coordinate, [0, -np.pi / 2],
                     [Sat.vsat * np.cos(Sat.azimuth), Sat.vsat * np.sin(Sat.azimuth), 0],
                     np.pi / 4, Ant_type='ULA', Num=32, Delta=c / fc / 2)

# 过信道
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw)
