import Antenna
import evolution


def non_stationary_channel(inputWaveform, Tx_Ant: Antenna, Rx_Ant: Antenna, fc, bw, isLOS: bool):
    # 平面波假设，Tx天线可见所有簇，Rx天线可见簇演进

    # 初始时刻，天线0上可见簇
    Rx_cluster_set = evolution.cluster_evolution_init(Tx_Ant, Rx_Ant)

    # 初始时刻在天线轴上的演进
    Rx_cluster_set = evolution.cluster_evolution_Ant(Rx_cluster_set, Rx_Ant)

    evolution.cluster_evolution_Ant_plot(Rx_cluster_set)

    # 该时刻下的信道系数
    # 计算NLOS分量
    for i in range(Rx_Ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_Ant.num):
            Clusters = Rx_cluster_set[i]

    if isLOS:
        pass

    return None
