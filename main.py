import commpy
import numpy as np

import Antenna
import channel_model

# 参数设置
n = int(10e6)  # 比特数
fc = 2e9  # 中心频率，单位Hz
bw = 100e6  # 信号带宽
c = 3e8  # 光速

# 生成发送信号
bits = np.random.binomial(1, 0.5, n)
qpsk = commpy.PSKModem(4)
x = qpsk.modulate(bits)

# 生成天线
Tx = Antenna.Antenna([0, 0, 300e3], [np.pi/2, -np.pi / 2], [0, 0.3, 0], Ant_type='ULA', Num=32, Delta=c/fc/2)
Rx = Antenna.Antenna([10, 0, 1.5], [0, np.pi/2], [0.5, 0, 0], Ant_type='ULA', Num=32, Delta=c/fc/2)

# 过信道
y = channel_model.non_stationary_channel(x, Tx, Rx, fc, bw)
