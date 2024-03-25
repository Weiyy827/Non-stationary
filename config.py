Re = 6356.9008e3  # 地球半径

bit_rate = int(10e6)  # bit速率10M
t = 1e-3  # 发送时间1ms
n = int(bit_rate*t)
fs = int(1e9)  # 采样率
fc = 2e9  # 中心频率，单位Hz
bw = 100e6  # 信号带宽
