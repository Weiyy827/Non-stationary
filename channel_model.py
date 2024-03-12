import copy

import numpy as np
import matplotlib.pyplot as plt

import Antenna
from cluster import cluster
from config import Lambda_R, N


def non_stationary_channel(inputWaveform, Tx_Ant:Antenna, Rx_Ant:Antenna, fc, bw, isLOS: bool):
    # 平面波假设，Tx天线可见所有簇，Rx天线可见簇演进

    # 初始时刻，天线0上可见簇

    Rx_cluster_set = []
    Ant_cluster_set = []
    for i in range(N):
        Ant_cluster_set.append(cluster(Tx_Ant, Rx_Ant, i))

    Rx_cluster_set.append(Ant_cluster_set)

    # 初始时刻在天线轴上的演进
    if Rx_Ant.Ant_type == 'ULA':
        Ds = 10  # 空间相关性参数，由场景决定，可选10，30，50，100
        P_survival = np.exp(-Lambda_R * Rx_Ant.delta_Ant / Ds)

        for i in range(1, Rx_Ant.num):
            temp = copy.deepcopy(Rx_cluster_set[i - 1])  # 复制列表到内存另一块
            for j in temp:
                if np.random.rand() < P_survival:
                    pass
                else:
                    temp.remove(j)
            Rx_cluster_set.append(temp)
    if Rx_Ant.Ant_type == 'URA':
        pass

    # 画出初始时刻簇在天线轴上的演进
    x_cord = []
    y_cord = []
    for Ant in Rx_cluster_set:
        for clusters in Ant:
            x_cord.append(Rx_cluster_set.index(Ant))
            y_cord.append(clusters.idx)

    plt.scatter(x_cord, y_cord)
    plt.xlabel("Antenna Index")
    plt.ylabel("Cluster")
    plt.yticks(np.arange(N))
    plt.title("t=0, Cluster Evolution on Antenna Axis")
    plt.show()

    # 该时刻下的信道系数
    # 计算NLOS分量
    for i in range(Rx_Ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_Ant.num):
            Clusters = Rx_cluster_set[i]

    if isLOS:
        pass

    return None
