import numpy as np

import antenna
import config
import evolution
import coeff_calculation


def non_stationary_channel(
    inputWaveform, Tx_ant: antenna.Antenna, Rx_ant: antenna.Antenna, fc, bw
):
    # 平面波假设，Tx天线可见所有簇，Rx天线可见簇演进

    # 初始时刻，天线0上可见簇
    Rx_cluster_set = evolution.cluster_evolution_init(Tx_ant, Rx_ant)

    # 初始时刻在天线轴上的演进
    Rx_cluster_set = evolution.cluster_evolution_Ant(Rx_cluster_set, Rx_ant)

    evolution.cluster_evolution_Ant_plot(Rx_cluster_set)

    # 该时刻下的信道系数
    # 计算小尺度衰落信道系数
    h = np.zeros([Rx_ant.num, Tx_ant.num], complex)
    # LOS分量
    # Rx场量
    vector_LOS_Rx = Tx_ant.position - Rx_ant.position
    azimuth_LOS_Rx = np.arctan(vector_LOS_Rx[1] / vector_LOS_Rx[0])
    elevation_LOS_Rx = np.arctan(
        vector_LOS_Rx[2] / np.sqrt(vector_LOS_Rx[1] ** 2 + vector_LOS_Rx[0] ** 2)
    )
    # Tx场量
    vector_LOS_Tx = Rx_ant.position - Tx_ant.position
    azimuth_LOS_Tx = np.arctan(vector_LOS_Tx[1] / vector_LOS_Tx[0])
    elevation_LOS_Tx = np.arctan(
        vector_LOS_Tx[2] / np.sqrt(vector_LOS_Tx[1] ** 2 + vector_LOS_Tx[0] ** 2)
    )

    #  交叉极化项
    phase = np.random.uniform(-np.pi, np.pi, 2)
    cross_polar = np.array(
        [np.exp(1j * phase[0]), 0, 0, -np.exp(1j * phase[1])]
    ).reshape([2, 2])

    faraday = np.array(
        [
            np.cos(108 / (fc**2)),
            -np.sin(108 / (fc**2)),
            np.sin(108 / (fc**2)),
            np.cos(108 / (fc**2)),
        ]
    ).reshape([2, 2])

    delay_LOS = (
        np.sqrt(vector_LOS_Tx[0] ** 2 + vector_LOS_Tx[1] ** 2 + vector_LOS_Tx[2] ** 2)
        / config.c
    )
    hL = (
        np.transpose(
            coeff_calculation.field(elevation_LOS_Rx, azimuth_LOS_Rx, Rx_ant.slant)
        )
        @ cross_polar
        @ faraday
        @ coeff_calculation.field(elevation_LOS_Tx, azimuth_LOS_Tx, Tx_ant.slant)
        * np.exp(1j * 2 * np.pi * fc * delay_LOS)
    )
    # NLOS分量

    for i in range(Rx_ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_ant.num):  # 第i个Rx天线对第j个Tx天线
            clusters = Rx_cluster_set[i]  # 第i个Rx天线上的可见簇集合
            hN = 0
            for k in clusters:  # 第k个可见簇
                hN_k = 0
                for m in range(k.sub):  # 簇内的子簇
                    #  1.用子簇的Rx A和E计算接收场量

                    # 0:Tx Azimuth; 1:Tx Elevation
                    # 2:Rx Azimuth; 3:Rx Elevation
                    Rx_field = coeff_calculation.field(
                        k.angle_sub[m][3], k.angle_sub[m][2], Rx_ant.slant
                    )

                    #  2.交叉极化项
                    cross_polar = coeff_calculation.cross_polar(
                        k.xnm_sub[m], k.phase_sub[m]
                    )

                    #  3.法拉第旋转，在载波频率小于10GHz时需要考虑

                    #  4.发射场量
                    Tx_field = coeff_calculation.field(
                        k.angle_sub[m][1], k.angle_sub[m][0], Tx_ant.slant
                    )

                    #  5.功率量
                    power = np.sqrt(k.power_sub[m])

                    #  6.多普勒量
                    doppler = np.exp(1j * 2 * np.pi * fc * k.absolute_delay)

                    #  7.时延项
                    #  时延滤波器，将时间时延转换为采样时延
                    hN_k += (
                        np.transpose(Rx_field)
                        @ cross_polar
                        @ faraday
                        @ Tx_field
                        * power
                        * doppler
                    )
                hN += hN_k
            K = 10 ** (3 / 10)  # k因子3dB
            h[i][j] = np.sqrt(K / (K + 1)) * hL + np.sqrt(1 / (K + 1)) * hN
    return None
