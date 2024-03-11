import commpy
import numpy as np

import channel_model

# 参数设置
n = 200000  # 比特数
numTx = 4  # 发送天线数
numRx = 4  # 接收天线数
fc = 60e9  # 中心频率，单位Hz

# 生成发送信号
bits = np.random.binomial(1, 0.5, n)
qpsk = commpy.PSKModem(4)
x = qpsk.modulate(bits)
# 过信道

y = channel_model.non_stationary_channel(x, numTx, numRx,fc)
