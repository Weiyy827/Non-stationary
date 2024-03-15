import numpy as np

import antenna
import evolution
import coeff_calculation


def non_stationary_channel(inputWaveform, Tx_Ant: antenna.Antenna, Rx_Ant: antenna.Antenna, fc, bw):
    # 平面波假设，Tx天线可见所有簇，Rx天线可见簇演进

    # 初始时刻，天线0上可见簇
    Rx_cluster_set = evolution.cluster_evolution_init(Tx_Ant, Rx_Ant)

    # 初始时刻在天线轴上的演进
    Rx_cluster_set = evolution.cluster_evolution_Ant(Rx_cluster_set, Rx_Ant)

    evolution.cluster_evolution_Ant_plot(Rx_cluster_set)

    # 该时刻下的信道系数
    # 计算小尺度衰落信道系数
    hN = np.zeros([Rx_Ant.num, Tx_Ant.num], complex)
    for i in range(Rx_Ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_Ant.num):  # 第i个Rx天线对第j个Tx天线
            Clusters = Rx_cluster_set[i]  # 第i个Rx天线上的可见簇集合
            hN_cluster = 0
            for k in Clusters:  # 第k个可见簇
                hN_k = 0
                for m in range(k.Sub):  # 簇内的子簇
                    #  1.用子簇的Rx A和E计算接收场量

                    # 0:Tx Azimuth; 1:Tx Elevation
                    # 2:Rx Azimuth; 3:Rx Elevation
                    Rx_field = coeff_calculation.field(k.Angle_sub[m][3], k.Angle_sub[m][2], Rx_Ant.slant)

                    #  2.交叉极化项
                    Cross_polar = coeff_calculation.cross_polar(k.Xnm_sub[m], k.Phase_sub[m])

                    #  3.法拉第旋转，在载波频率小于10GHz时需要考虑
                    Faraday = np.array([np.cos(108 / (fc ** 2)), -np.sin(108 / (fc ** 2)),
                                        np.sin(108 / (fc ** 2)), np.cos(108 / (fc ** 2))]).reshape([2, 2])
                    #  4.发射场量
                    Tx_field = coeff_calculation.field(k.Angle_sub[m][1], k.Angle_sub[m][0], Tx_Ant.slant)

                    #  5.功率量
                    Power = np.sqrt(k.Power_sub[m])

                    #  6.多普勒量
                    Doppler = np.exp(1j * 2 * np.pi * k.Delay)

                    #  7.时延项
                    #  时延滤波器，将时间时延转换为采样时延
                    hN_k += np.transpose(Rx_field) @ Cross_polar @ Faraday @ Tx_field * Power * Doppler
                hN_cluster += hN_k
            hN[i][j] = hN_cluster
    return None
