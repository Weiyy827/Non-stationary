Re = 6356.9008e3  # 地球半径

bit_rate = int(10e9)  # bit速率10Gbps
t = 1e-6  # 发送时间10us
n = int(bit_rate*t)
fs = int(10e9)  # 采样率10GHz
fc = 2e9  # 中心频率，单位Hz
bw = 100e6  # 信号带宽
