import numpy as np

import antenna
import evolution
import coeff_calculation


def non_stationary_channel(inputWaveform, Tx_Ant: antenna, Rx_Ant: antenna, fc, bw):
    # 平面波假设，Tx天线可见所有簇，Rx天线可见簇演进

    # 初始时刻，天线0上可见簇
    Rx_cluster_set = evolution.cluster_evolution_init(Tx_Ant, Rx_Ant)

    # 初始时刻在天线轴上的演进
    Rx_cluster_set = evolution.cluster_evolution_Ant(Rx_cluster_set, Rx_Ant)

    evolution.cluster_evolution_Ant_plot(Rx_cluster_set)

    # 该时刻下的信道系数
    # 计算小尺度衰落信道系数
    for i in range(Rx_Ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_Ant.num):
            Clusters = Rx_cluster_set[i]  # 第i个Rx天线上的可见簇集合
            for k in Clusters:  # 第k个可见簇
                for m in range(k.Mn):  # 簇内的子簇
                    #  1.用子簇的Rx A和E计算接收场量
                    # 0:Tx Azimuth; 1:Tx Elevation
                    # 2:Rx Azimuth; 3:Rx Elevation

                    Rx_field = coeff_calculation.rx_field(k.Angle_Mn[m][2], k.Angle_Mn[m][3])
                    #  2.交叉极化项
                    Cross_polar = coeff_calculation.cross_polar()
                    #  3.发射场量
                    Tx_field = coeff_calculation.tx_field(k.Angle_Mn[m][1], k.Angle_Mn[m][0])
                    #  4.功率量
                    Power = np.sqrt(k.Power_Mn[m])
                    #  5.多普勒量
                    Doppler = np.exp(1j * 2 * np.pi * k.Delay[m])
                    #  6.时延项

    return None
