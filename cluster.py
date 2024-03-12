import numpy as np

import Antenna


class cluster:
    def __init__(self, Tx_Ant: Antenna, Rx_Ant: Antenna):
        # 簇参数设置

        self.Power = None
        self.Power_Mn = []

        self.Delay = None

        self.Angle = []
        self.Angle_Mn = []

        self.Mn = None

        # 天线设置
        self.Tx_Ant = Tx_Ant
        self.Rx_Ant = Rx_Ant

        #  生成一个新簇
        #  生成簇内的子簇
        self.Mn = np.max([np.random.poisson(20), 1])

        #  生成簇时延
        Un = np.random.rand()
        Rtau = 2.3  # 时延因子 2.3：NLOS城市室外，2.4：NLOS办公室室内
        Stau = np.power(10, 0.32) * np.random.randn() + np.power(10, -6.63)  # 随机生成时延扩展
        self.Delay = -Rtau * Stau * np.log(Un)

        #  生成簇平均功率
        Zn = np.sqrt(3) * np.random.randn()
        self.Power = np.exp(-self.Delay * (Rtau - 1) / (Rtau * Stau)) * np.power(10, -Zn / 10)

        #  生成簇AOA,AOD
        self.Angle.append(1.15 * np.random.randn() + self.Tx_Ant.azimuth)
        self.Angle.append(0.18 * np.random.randn() + self.Tx_Ant.elevation)
        self.Angle.append(0.54 * np.random.randn() + self.Rx_Ant.azimuth)
        self.Angle.append(0.11 * np.random.randn() + self.Rx_Ant.elevation)

        #  生成子簇平均功率
        for i in range(self.Mn):
            Znm = np.sqrt(3) * np.random.randn()
            self.Power_Mn.append(np.exp(1 - Rtau) * np.power(10, -Znm / 10))

        #  生成子簇角度参数
        for i in range(self.Mn):
            Delta_angle = np.random.laplace(0, 0.017, 4)
            self.Angle_Mn.append(self.Angle + Delta_angle)

        #  簇被哪些子天线可见

    def cluster_update(self, time_interval):
        # Δt时刻后
        Pc = 0.3  # 运动散射簇的比例
        vrx = 1  # 接收端移动速度
        vct, vcr = 0, 0.5  # 散射簇相对于发，收天线的速度
        # 新增的散射簇
        delta = Pc * (vct + vcr) * Delta_t + vrx * Delta_t

        new_cluster = np.floor(Lambda_G / Lambda_R * (1 - np.exp(-Lambda_R * delta / Ds)))
        P_survival_time = np.exp(-Lambda_R * delta / Ds)

        # 时间轴上的演进
        for k in range(1, int(t / Delta_t)):
            N += int(new_cluster)
            C.append(np.zeros([numTx, N]))
            C[k][0] = np.arange(1, N + 1)
            for i in range(1, numTx):
                idx = 0
                for j in C[k][i - 1]:
                    if j != 0:
                        if np.random.rand() < P_survival_time:
                            C[k][i][idx] = j
                        else:
                            C[k][i][idx] = 0
                    idx += 1
        return C


if __name__ == '__main__':
    Tx = Antenna.Antenna([1, 32], [0, 0, 0], [0, np.pi / 4], 0.1)
    Rx = Antenna.Antenna([1, 32], [10, 0, 0], [np.pi, 0], 0.1)
    Cluster = cluster(Tx, Rx)
