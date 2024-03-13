import numpy as np

import antenna
from config import Rtau, Stau


class cluster:
    def __init__(self, Tx_Ant: Antenna, Rx_Ant: Antenna, idx: int):
        # 簇参数设置
        self.idx = idx

        self.Power = None
        self.Power_Mn = []

        self.Delay = None

        self.Angle = []
        self.Angle_Mn = []

        self.Mn = None

        self.Position_Rx = np.array([])
        self.Position_Tx = np.array([])
        self.Position_Mn_Rx = []
        self.Position_Mn_Tx = []

        # 天线设置
        self.Tx_Ant = Tx_Ant
        self.Rx_Ant = Rx_Ant

        #  生成一个新簇
        #  生成簇内的子簇
        self.Mn = np.max([np.random.poisson(20), 1])

        #  生成簇时延
        Un = np.random.rand()

        self.Delay = -Rtau * Stau * np.log(Un)

        #  生成簇平均功率
        Zn = np.sqrt(3) * np.random.randn()
        self.Power = np.exp(-self.Delay * (Rtau - 1) / (Rtau * Stau)) * np.power(10, -Zn / 10)

        #  生成簇AOA,AOD
        self.Angle.append(1.15 * np.random.randn() + self.Tx_Ant.azimuth)
        self.Angle.append(0.18 * np.random.randn() + self.Tx_Ant.elevation)
        self.Angle.append(0.54 * np.random.randn() + self.Rx_Ant.azimuth)
        self.Angle.append(0.11 * np.random.randn() + self.Rx_Ant.elevation)

        #  生成簇的位置
        d = np.sqrt((Tx_Ant.position[0] - Rx_Ant.position[0]) ** 2 +
                    (Tx_Ant.position[1] - Rx_Ant.position[1]) ** 2 +
                    (Tx_Ant.position[2] - Rx_Ant.position[2]) ** 2)
        D = np.array([d, 0, 0])

        self.Position_Rx = (np.sqrt(15) * np.random.randn() + 25) * np.array(
            [np.cos(self.Angle[3]) * np.cos(self.Angle[2]),
             np.cos(self.Angle[3]) * np.sin(self.Angle[2]),
             np.sin(self.Angle[3])])

        self.Position_Tx = (np.sqrt(10) * np.random.randn() + 30) * np.array(
            [np.cos(self.Angle[1]) * np.cos(self.Angle[0]),
             np.cos(self.Angle[1]) * np.sin(self.Angle[0]),
             np.sin(self.Angle[1])]) + D

        #  生成子簇平均功率
        for i in range(self.Mn):
            Znm = np.sqrt(3) * np.random.randn()
            self.Power_Mn.append(np.exp(1 - Rtau) * np.power(10, -Znm / 10))

        #  生成子簇角度参数
        for i in range(self.Mn):
            Delta_angle = np.random.laplace(0, 0.017, 4)
            self.Angle_Mn.append(self.Angle + Delta_angle)

        #  生成子簇的位置矢量
        for i in range(self.Mn):
            self.Position_Mn_Rx.append(
                (np.sqrt(15) * np.random.randn() + 25) * np.array(
                    [np.cos(self.Angle_Mn[i][3]) * np.cos(self.Angle_Mn[i][2]),
                     np.cos(self.Angle_Mn[i][3]) * np.sin(self.Angle_Mn[i][2]),
                     np.sin(self.Angle_Mn[i][3])]))

            self.Position_Mn_Tx.append(
                (np.sqrt(15) * np.random.randn() + 25) * np.array(
                    [np.cos(self.Angle_Mn[i][1]) * np.cos(self.Angle_Mn[i][0]),
                     np.cos(self.Angle_Mn[i][1]) * np.sin(self.Angle_Mn[i][0]),
                     np.sin(self.Angle_Mn[i][1])]) + D)
