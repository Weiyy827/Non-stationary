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
Rx = antenna.Antenna([10, 0, 1.5], [0, np.pi / 2], [0.5, 0, 0], np.pi / 4, Ant_type='ULA', Num=32,
                     Delta=c / fc / 2)
Tx = antenna.Antenna(
    [Sat.height / np.tan(Sat.elevation) * np.cos(Sat.azimuth),
     Sat.height / np.tan(Sat.elevation) * np.sin(Sat.azimuth), Sat.height],
    [np.pi / 2, -np.pi / 2], [Sat.vsat * np.cos(Sat.azimuth), Sat.vsat * np.sin(Sat.azimuth), 0],
    np.pi / 4, Ant_type='ULA', Num=32, Delta=c / fc / 2)

# 过信道
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw)
