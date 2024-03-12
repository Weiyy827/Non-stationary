import commpy
import numpy as np

import Antenna
import channel_model
from config import fc, bw, n, c, isLOS

# 参数设置

# 生成发送信号
bits = np.random.binomial(1, 0.5, n)
qpsk = commpy.PSKModem(4)
x = qpsk.modulate(bits)

# 生成天线
Tx = Antenna.Antenna([0, 0, 5], [np.pi/2, -np.pi / 2], [0, 0.3, 0], Ant_type='ULA', Num=32, Delta=c/fc/2)
Rx = Antenna.Antenna([10, 0, 1.5], [0, np.pi/2], [0.5, 0, 0], Ant_type='ULA', Num=32, Delta=c/fc/2)

# 过信道
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw, isLOS)
