import numpy as np

import antenna
import config


class Cluster:
    def __init__(self, Tx_ant: antenna.Antenna, Rx_ant: antenna.Antenna, idx: int):
        # 簇参数设置

        self.idx = idx

        self.power_sub = []

        self.angle = []
        self.angle_sub = []

        self.position_sub = []
        self.vector_sub_Tx = []
        self.vector_sub_Rx = []

        self.xnm_sub = []
        self.phase_sub = []

        # 天线设置
        self.Tx_ant = Tx_ant
        self.Rx_ant = Rx_ant

        #  生成一个新簇
        #  生成簇内的子簇
        self.sub = np.max([np.random.poisson(20), 1])

        #  生成簇AOA,AOD
        self.angle.append(1.15 * np.random.randn() + self.Tx_ant.azimuth)
        self.angle.append(0.18 * np.random.randn() + self.Tx_ant.elevation)

        self.angle.append(0.54 * np.random.randn() + self.Rx_ant.azimuth)
        self.angle.append(0.11 * np.random.randn() + self.Rx_ant.elevation)

        #  生成簇的位置，本地GCS
        self.position = (np.sqrt(15) * np.random.randn() + 25) * np.array(
            [
                np.cos(self.angle[3]) * np.cos(self.angle[2]),
                np.cos(self.angle[3]) * np.sin(self.angle[2]),
                np.sin(self.angle[3]),
            ]
        ) + Rx_ant.position

        #  生成簇时延
        distance_Rx = np.sqrt(
            (Rx_ant.position[0] - self.position[0]) ** 2
            + (Rx_ant.position[1] - self.position[1]) ** 2
            + (Rx_ant.position[2] - self.position[2]) ** 2
        )
        distance_Tx = np.sqrt(
            (Tx_ant.position[0] - self.position[0]) ** 2
            + (Tx_ant.position[1] - self.position[1]) ** 2
            + (Tx_ant.position[2] - self.position[2]) ** 2
        )
        distance_LOS = np.sqrt(
            (Rx_ant.position[0] - Tx_ant.position[0]) ** 2
            + (Rx_ant.position[1] - Tx_ant.position[1]) ** 2
            + (Rx_ant.position[2] - Tx_ant.position[2]) ** 2
        )
        self.relative_delay = (distance_Rx + distance_Tx - distance_LOS) / config.c
        self.absolute_delay = (distance_Rx + distance_Tx) / config.c

        #  生成簇平均功率

        #  生成子簇的参数
        for i in range(self.sub):
            #  生成子簇极化交叉比
            self.xnm_sub.append(3 * np.random.randn() + 9)  # 暂时只考虑UMi

            #  生成子簇相位
            phase = np.random.uniform(-np.pi, np.pi, 4)
            self.phase_sub.append(phase)

            #  生成子簇平均功率，直接对应时延功率谱
            znm = np.sqrt(3) * np.random.randn()

            f = config.fc
            self.power_sub.append(
                np.exp(1 - Rtau) * np.power(10, -znm / 10) * (f / config.fc) ** 1
            )  # gamma = 1,f为频率

            #  生成子簇角度参数
            delta_angle = np.random.laplace(0, 0.017, 4)
            self.angle_sub.append(self.angle + delta_angle)

            #  生成子簇的位置矢量
            self.position_sub.append(
                (np.sqrt(15) * np.random.randn() + 25)
                * np.array(
                    [
                        np.cos(self.angle_sub[i][3]) * np.cos(self.angle_sub[i][2]),
                        np.cos(self.angle_sub[i][3]) * np.sin(self.angle_sub[i][2]),
                        np.sin(self.angle_sub[i][3]),
                    ]
                )
                + Rx_ant.position
            )
