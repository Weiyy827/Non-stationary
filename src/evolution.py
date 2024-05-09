import copy
import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.config import lambda_R, Ds
from src.simpar import Antenna


def cluster_evolution_Ant(cluster_number, ant: Antenna):
    """簇在天线轴上的演进

    Args:
        cluster_number (int): 簇数
        ant (Antenna): 目标天线

    Returns:
        list[list[Cluster]]: 演进完后天线上所有阵元的簇集合
    """
    if cluster_number < 10:
        ant_cluster = []
        for i in range(ant.num):
            temp = random.sample([i for i in range(cluster_number)], random.randint(0, 3))
            ant_cluster.append(temp)
    else:
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
            # TODO 完成URA在天线轴上的演进
            pass

    return ant_cluster


def cluster_evolution_Time():
    """簇在时间轴上的演进

    Returns:
        _type_: _description_
    """
    return None


def cluster_evolution_Ant_plot(cluster_set):
    """画出簇在天线轴上的演进图

    Args:
        cluster_set (list[list[Cluster]]): 演进后天线上可视的簇集合
    """
    # 画出初始时刻簇在天线轴上的演进
    x_cord = []
    y_cord = []
    for ant in cluster_set:
        for clusters in ant:
            x_cord.append(cluster_set.index(ant))
            y_cord.append(clusters)

    plt.scatter(x_cord, y_cord)
    plt.xlabel("Antenna Index")
    plt.ylabel("Cluster")
    plt.yticks(np.arange(3))
    plt.title("t=0, Cluster Evolution on Antenna Axis")
    matplotlib.use("TkAgg")
    plt.show()
