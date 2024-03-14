import numpy as np

n = int(10e6)  # 比特数
fs = 10e5  # 采样率
fc = 2e9  # 中心频率，单位Hz
bw = 100e6  # 信号带宽
c = 3e8  # 光速
isLOS = True

N = 20  # 初始可见簇
Lambda_G = 80  # 簇生成率
Lambda_R = 4  # 簇消亡率

Rtau = 2.3  # 时延因子 2.3：NLOS城市室外，2.4：NLOS办公室室内
Stau = np.power(10, 0.32) * np.random.randn() + np.power(10, -6.63)  # 随机生成时延扩展

