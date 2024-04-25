import matplotlib.pyplot as plt
import numpy as np
import scipy

from LSP import generate_lsp
from cluster import generate_cluster_param, Cluster, plot_PDP
from coeff_calculation import field, cross_polar
from config import lambda_G, lambda_R, Ds, Dt, dt
from evolution import cluster_evolution_Ant_plot, cluster_evolution_Ant
from simpar import Simulation_parameter
from utils import calculate_LOS_angle, gcs2lcs


def non_stationary_channel(
        simpar: Simulation_parameter
):
    """
    计算信道系数

    :param simpar：仿真场景参数
    :return: 信道系数
    """
    Tx_ant = simpar.tx
    Rx_ant = simpar.rx
    fc = simpar.fc
    snapshots = simpar.snapshots

    # 每个快照时刻更新
    for snapshot in range(snapshots):
        # 0.确定该时刻LOS径的到达角和离开角。方位角(0,2pi),天顶角(0，pi)
        LOS_angle = calculate_LOS_angle(Tx_ant, Rx_ant)

        # 1.计算该时刻大尺度衰落参数,单位log10(s),dB,log10(deg)，参考quadriga
        lsp = generate_lsp(Tx_ant, Rx_ant, fc)

        # 2.生成所有簇的参数，参考38.901
        cluster_number = 4
        cluster_param = generate_cluster_param(lsp, cluster_number, LOS_angle)
        cluster_set = []
        for cluster_idx in range(cluster_number):
            cluster_set.append(Cluster(lsp, cluster_param[cluster_idx]))

        plot_PDP(cluster_set)

    # 在非初始时刻，需要考虑簇的生灭过程
    # 以平稳时间为单位演进
    if snapshots:
        cluster_set_old = cluster_set
        for t in range(snapshots):
            # 更新天线位置
            Tx_ant.evolve()
            Rx_ant.evolve()

            # 计算此刻的LOS径角度
            LOS_angle = calculate_LOS_angle(Tx_ant, Rx_ant)

            # 计算此刻大尺度衰落
            lsp = generate_lsp(Tx_ant, Rx_ant, fc)

            # 计算该时刻新生簇数量
            Tx_speed = np.sqrt(np.sum(Tx_ant.velocity ** 2))
            Rx_speed = np.sqrt(np.sum(Rx_ant.velocity ** 2))
            delta = 0.3 * Tx_speed * dt + Rx_speed * dt
            cluster_number_new = round(lambda_G / lambda_R * (1 - np.exp(-lambda_R * delta / Ds)))

            # 生成新簇
            cluster_param_new = generate_cluster_param(lsp, cluster_number_new, LOS_angle)
            cluster_set_new = generate_cluster(cluster_param_new)

            # 旧簇死亡
            p_survival_time = np.exp(-lambda_R * Rx_speed * dt / Dt)
            for cluster in cluster_set_old:
                if np.random.rand() > p_survival_time:
                    cluster_set_old.remove(cluster)

            # 将两个簇集合合在一起，之后进行天线轴上的演进
            cluster_set = cluster_set_old + cluster_set_new

    else:
        cluster_set = cluster_set_init

    # 画出簇的时延功率谱
    delays = [i.delay for i in cluster_set]
    powers = [10 * np.log10(i.power) for i in cluster_set]
    plt.scatter(delays, powers)
    plt.xlabel("Delay[s]")
    plt.ylabel("Power[dB]")
    plt.title("Power-Delay Profile")
    plt.show()

    # 2.计算该时刻的簇演进
    # 在天线轴上进行演进
    Rx_cluster_set = cluster_evolution_Ant(cluster_set, Rx_ant)
    # 画出初始时刻天线上簇的集合
    cluster_evolution_Ant_plot(Rx_cluster_set)

    # 4. 该时刻下的信道系数
    # 计算小尺度衰落信道系数

    # 计算场量

    LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD = LOS_angle
    # 将角度转换到天线为原点的坐标系上
    LOS_AOA_LCS, LOS_ZOA_LCS = gcs2lcs(LOS_AOA, LOS_ZOA, Rx_ant)

    # 方法相同计算Tx天线为原点的LCS坐标
    LOS_AOD_LCS, LOS_ZOD_LCS = gcs2lcs(LOS_AOD, LOS_ZOD, Tx_ant)

    # 计算LOS径的三维方向矢量
    r_rx_LOS = np.array(
        [np.sin(LOS_ZOA) * np.cos(LOS_AOA), np.sin(LOS_ZOA) * np.sin(LOS_AOA),
         np.cos(LOS_ZOA)]).reshape([3, 1])

    r_tx_LOS = np.array(
        [np.sin(LOS_ZOD) * np.cos(LOS_AOD), np.sin(LOS_ZOD) * np.sin(LOS_AOD),
         np.cos(LOS_ZOD)]).reshape([3, 1])

    # 计算多普勒量
    vec = Tx_ant.position - Rx_ant.position
    doppler_1 = np.exp(-1j * 2 * np.pi * (np.sqrt(np.sum(vec ** 2))) / (scipy.constants.c / fc))
    doppler_2 = np.exp(1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.position) / (scipy.constants.c / fc))
    doppler_3 = np.exp(1j * 2 * np.pi * (r_tx_LOS.T @ Tx_ant.position) / (scipy.constants.c / fc))
    doppler_4 = np.exp(
        1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * snapshots)

    # 计算LOS径的信道系数,LOS径的时延设置为0
    hL = (
            np.transpose(
                field(LOS_ZOA_LCS, LOS_AOA_LCS, Rx_ant.slant)
            )
            @ np.array([1, 0, 0, -1]).reshape([2, 2])
            @ field(LOS_ZOD_LCS, LOS_AOD_LCS, Tx_ant.slant)
            * doppler_1
            * doppler_2
            * doppler_3
            * doppler_4
    )

    # NLOS分量信道系数

    # 对于第i个接收阵元
    for i in range(Rx_ant.num):

        # 对第j个发射阵元
        for j in range(Tx_ant.num):

            # 取出第i个接收阵元上的可见簇集合
            clusters = Rx_cluster_set[i]

            # 对于集合中的第n个簇
            for cluster in clusters:

                # 储存第n个簇的信道系数
                hN = 0

                # 对于簇内的第m个子簇
                for m in range(cluster.number):
                    #  1.用子簇的AOA和ZOA计算接收场量
                    ray_cAOA_LCS, ray_cZOA_LCS = gcs2lcs(cluster.cAOA[m], cluster.cZOA[m], Rx_ant)
                    Rx_field = field(ray_cZOA_LCS, ray_cAOA_LCS, Rx_ant.slant)

                    #  2.交叉极化项
                    polar = cross_polar(cluster.kappa[m], cluster.phase[m])

                    #  3.法拉第旋转，在载波频率小于10GHz时需要考虑
                    faraday = np.array([np.cos(108 / (fc ** 2)), -np.sin(108 / (fc ** 2)),
                                        np.sin(108 / (fc ** 2)), np.cos(108 / (fc ** 2))]).reshape([2, 2])

                    #  4.发射场量
                    ray_cAOD_LCS, ray_cZOD_LCS = gcs2lcs(cluster.cAOD[m], cluster.cZOD[m], Rx_ant)
                    Tx_field = field(ray_cZOD_LCS, ray_cAOD_LCS, Tx_ant.slant)

                    #  5.多普勒量
                    r_rx = np.array(
                        [np.sin(cluster.cZOA[m]) * np.cos(cluster.cAOA[m]),
                         np.sin(cluster.cZOA[m]) * np.sin(cluster.cAOA[m]),
                         np.cos(cluster.cZOA[m])]).reshape([3, 1])
                    r_tx = np.array(
                        [np.sin(cluster.cZOD[m]) * np.cos(cluster.cAOD[m]),
                         np.sin(cluster.cZOD[m]) * np.sin(cluster.cAOD[m]),
                         np.cos(cluster.cZOD[m])]).reshape([3, 1])

                    doppler_1 = np.exp(1j * 2 * np.pi * (r_rx.T @ Rx_ant.position) / (scipy.constants.c / fc))
                    doppler_2 = np.exp(1j * 2 * np.pi * (r_tx.T @ Tx_ant.position) / (scipy.constants.c / fc))
                    doppler_3 = np.exp(
                        1j * 2 * np.pi * (r_rx.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * snapshots)

                    # 子簇的信道系数之和为这个簇的信道系数
                    hN += (
                            np.sqrt(cluster.power / cluster.number)
                            * np.transpose(Rx_field)
                            @ polar
                            @ faraday
                            @ Tx_field
                            * doppler_1
                            * doppler_2
                            * doppler_3
                    )
    # 返回输出波形
    return outputWaveform
