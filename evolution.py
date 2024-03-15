import copy

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import antenna
from cluster import Cluster
from config import N, Lambda_R


def cluster_evolution_init(Tx_ant: antenna.Antenna, Rx_ant: antenna.Antenna):
    cluster_set = []
    ant_cluster_set = []
    for i in range(N):
        ant_cluster_set.append(Cluster(Tx_ant, Rx_ant, i))

    cluster_set.append(ant_cluster_set)

    return cluster_set


def cluster_evolution_Ant(cluster_set, ant: antenna.Antenna):
    if ant.ant_type == "ULA":
        Ds = 10  # 空间相关性参数，由场景决定，可选10，30，50，100
        P_survival = np.exp(-Lambda_R * ant.ant_spacing / Ds)

        for i in range(1, ant.num):
            temp = copy.deepcopy(cluster_set[i - 1])  # 复制列表到内存另一块
            for j in temp:
                if np.random.rand() < P_survival:
                    pass
                else:
                    temp.remove(j)
            cluster_set.append(temp)

    if ant.ant_type == "URA":
        pass

    return cluster_set


def cluster_evolution_Time():
    return None


def cluster_evolution_Ant_plot(cluster_set):
    # 画出初始时刻簇在天线轴上的演进
    x_cord = []
    y_cord = []
    for ant in cluster_set:
        for clusters in ant:
            x_cord.append(cluster_set.index(ant))
            y_cord.append(clusters.idx)

    plt.scatter(x_cord, y_cord)
    plt.xlabel("Antenna Index")
    plt.ylabel("Cluster")
    plt.yticks(np.arange(N))
    plt.title("t=0, Cluster Evolution on Antenna Axis")
    matplotlib.use("TkAgg")
    plt.show()
