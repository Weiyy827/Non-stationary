import numpy as np

import antenna
import config
from config import Rtau, Stau


class cluster:
    def __init__(self, Tx_Ant: antenna.Antenna, Rx_Ant: antenna.Antenna, idx: int):
        # 簇参数设置

        self.idx = idx

        self.Power_sub = []

        self.Angle = []
        self.Angle_sub = []

        self.Position_sub = []
        self.V_Position_sub_Tx = []
        self.V_Position_sub_Rx = []

        self.Xnm_sub = []
        self.Phase_sub = []

        # 天线设置
        self.Tx_Ant = Tx_Ant
        self.Rx_Ant = Rx_Ant

        #  生成一个新簇
        #  生成簇内的子簇
        self.Sub = np.max([np.random.poisson(20), 1])

        #  生成簇AOA,AOD
        self.Angle.append(1.15 * np.random.randn() + self.Tx_Ant.azimuth)
        self.Angle.append(0.18 * np.random.randn() + self.Tx_Ant.elevation)

        self.Angle.append(0.54 * np.random.randn() + self.Rx_Ant.azimuth)
        self.Angle.append(0.11 * np.random.randn() + self.Rx_Ant.elevation)

        #  生成簇的位置，本地GCS
        self.Position = (np.sqrt(15) * np.random.randn() + 25) * np.array(
            [np.cos(self.Angle[3]) * np.cos(self.Angle[2]),
             np.cos(self.Angle[3]) * np.sin(self.Angle[2]),
             np.sin(self.Angle[3])]) + Rx_Ant.position

        #  生成簇相对LOS的时延
        Distance_Rx = np.sqrt((Rx_Ant.position[0] - self.Position[0]) ** 2 + (Rx_Ant.position[1] - self.Position[1]) ** 2 + (Rx_Ant.position[2] - self.Position[2]) ** 2)
        Distance_Tx = np.sqrt((Tx_Ant.position[0] - self.Position[0]) ** 2 + (Tx_Ant.position[1] - self.Position[1]) ** 2 + (Tx_Ant.position[2] - self.Position[2]) ** 2)
        Distance_LOS = np.sqrt((Rx_Ant.position[0] - Tx_Ant.position[0]) ** 2 + (Rx_Ant.position[1] - Tx_Ant.position[1]) ** 2 + (Rx_Ant.position[2] - Tx_Ant.position[2]) ** 2)
        self.Delay = (Distance_Rx+Distance_Tx-Distance_LOS)/config.c

        #  生成簇平均功率
        Zn = np.sqrt(3) * np.random.randn()
        self.Power = np.exp(-self.Delay * (Rtau - 1) / (Rtau * Stau)) * np.power(10, -Zn / 10)

        #  生成子簇的参数
        for i in range(self.Sub):
            #  生成子簇极化交叉比
            self.Xnm_sub.append(3 * np.random.randn() + 9)  # 暂时只考虑UMi

            #  生成子簇相位
            Phase = np.random.uniform(-np.pi, np.pi, 4)
            self.Phase_sub.append(Phase)

            #  生成子簇平均功率
            Znm = np.sqrt(3) * np.random.randn()
            self.Power_sub.append(np.exp(1 - Rtau) * np.power(10, -Znm / 10))

            #  生成子簇角度参数
            Delta_angle = np.random.laplace(0, 0.017, 4)
            self.Angle_sub.append(self.Angle + Delta_angle)

            #  生成子簇的位置矢量
            self.Position_sub.append(
                (np.sqrt(15) * np.random.randn() + 25) * np.array(
                    [np.cos(self.Angle_sub[i][3]) * np.cos(self.Angle_sub[i][2]),
                     np.cos(self.Angle_sub[i][3]) * np.sin(self.Angle_sub[i][2]),
                     np.sin(self.Angle_sub[i][3])]) + Rx_Ant.position)
