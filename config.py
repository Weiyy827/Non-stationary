Re = 6356.9008e3  # 地球半径

bit_rate = int(10e9)  # bit速率10Gbps
t = 1e-6  # 发送时间10us
dt = 1e-3  # 时间单位
n = int(bit_rate * t)
fs = int(10e9)  # 采样率10GHz
fc = 2e9  # 中心频率，单位Hz
bw = 100e6  # 信号带宽

lambda_G = 80  # 簇生成率
lambda_R = 4  # 簇死亡率
Ds = 10  # 距离相关参数
Dt = 10  # 时间相关参数
