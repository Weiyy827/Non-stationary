import copy

import numpy as np
from matplotlib import pyplot as plt

from cluster import cluster
from config import N, Lambda_R


def cluster_evolution_init(Tx_Ant, Rx_Ant):
    Cluster_set = []
    Ant_cluster_set = []
    for i in range(N):
        Ant_cluster_set.append(cluster(Tx_Ant, Rx_Ant, i))

    Cluster_set.append(Ant_cluster_set)

    return Cluster_set


def cluster_evolution_Ant(Cluster_set, Ant):
    if Ant.Ant_type == 'ULA':
        Ds = 10  # 空间相关性参数，由场景决定，可选10，30，50，100
        P_survival = np.exp(-Lambda_R * Ant.delta_Ant / Ds)

        for i in range(1, Ant.num):
            temp = copy.deepcopy(Cluster_set[i - 1])  # 复制列表到内存另一块
            for j in temp:
                if np.random.rand() < P_survival:
                    pass
                else:
                    temp.remove(j)
            Cluster_set.append(temp)

    if Ant.Ant_type == 'URA':
        pass

    return Cluster_set


def cluster_evolution_Time():
    return None


def cluster_evolution_Ant_plot(Cluster_set):
    # 画出初始时刻簇在天线轴上的演进
    x_cord = []
    y_cord = []
    for Ant in Cluster_set:
        for clusters in Ant:
            x_cord.append(Cluster_set.index(Ant))
            y_cord.append(clusters.idx)

    plt.scatter(x_cord, y_cord)
    plt.xlabel("Antenna Index")
    plt.ylabel("Cluster")
    plt.yticks(np.arange(N))
    plt.title("t=0, Cluster Evolution on Antenna Axis")
    plt.show()
