import copy

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import scenario
from config import lambda_R, Ds


def cluster_evolution_Ant(cluster_set, ant: scenario.Antenna):
    """
    簇在天线轴上的演进

    :param cluster_set: 天线上所有阵元的簇集合
    :param ant: 天线对象
    :return: 演进完后天线上所有阵元的簇集合
    """
    if ant.ant_type == "ULA":
        p_survival = np.exp(-lambda_R * ant.ant_spacing / Ds)
        ant_cluster = [cluster_set]
        for i in range(1, ant.num):
            temp = copy.deepcopy(ant_cluster[i - 1])  # 复制列表到内存另一块
            for j in temp:
                if np.random.rand() < p_survival:
                    pass
                else:
                    temp.remove(j)
            ant_cluster.append(temp)

    if ant.ant_type == "URA":
        pass

    return ant_cluster


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
    plt.yticks(np.arange(20))
    plt.title("t=0, Cluster Evolution on Antenna Axis")
    matplotlib.use("TkAgg")
    plt.show()
