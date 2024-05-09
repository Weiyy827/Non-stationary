"""计算信道系数并储存在csv文件中
"""

import csv
import numpy as np

from src.LSP_3GPP import generate_lsp_3GPP
from src.LSP_quadriga import generate_lsp_quadriga
from src.cluster import generate_cluster_param, Cluster, plot_PDP
from src.coeff_calculation import generate_LOS_component, generate_NLOS_component
from src.evolution import cluster_evolution_Ant, cluster_evolution_Ant_plot
from src.simpar import SimulationParameter
from src.utils import calculate_LOS_angle


def non_stationary_channel(simpar: SimulationParameter) -> tuple[list, list]:
    """
    计算当前仿真参数下的信道系数

    :param simpar：仿真场景参数
    :return: 信道系数
    """

    # 一、取出仿真参数
    Tx_ant = simpar.tx
    Rx_ant = simpar.rx
    fc = simpar.fc
    snapshots = simpar.snapshots

    # 二、计算每个快照时刻的信道系数
    for snapshot in range(snapshots):

        # 0.计算该时刻LOS径的到达角和离开角
        LOS_angle = calculate_LOS_angle(Tx_ant, Rx_ant)

        # 1.参考quadriga或38.811的方式，生产大尺度参数
        generate_method = "3GPP"
        if generate_method == "3GPP":
            lsp = generate_lsp_3GPP(Tx_ant, Rx_ant, simpar)

        elif generate_method == "quadriga":
            lsp = generate_lsp_quadriga(Tx_ant, Rx_ant, fc)

        # 2.生成所有簇的参数，参考38.901公式
        cluster_number = 3
        cluster_param = generate_cluster_param(
            lsp, cluster_number, LOS_angle, simpar.isLOS
        )
        cluster_set = []
        for cluster_idx in range(cluster_number):
            cluster_set.append(Cluster(lsp, cluster_param[cluster_idx], cluster_idx))

        plot_PDP(cluster_set)

    # 在非初始时刻，需要考虑簇的生灭过程
    # 以平稳时间为单位演进

    # TODO 完成时间演进，可能需要重构模块
    if snapshots - 1:
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
            Tx_speed = np.sqrt(np.sum(Tx_ant.velocity**2))
            Rx_speed = np.sqrt(np.sum(Rx_ant.velocity**2))
            delta = 0.3 * Tx_speed * dt + Rx_speed * dt
            cluster_number_new = round(
                lambda_G / lambda_R * (1 - np.exp(-lambda_R * delta / Ds))
            )

            # 生成新簇
            cluster_param_new = generate_cluster_param(
                lsp, cluster_number_new, LOS_angle
            )
            cluster_set_new = generate_cluster(cluster_param_new)

            # 旧簇死亡
            p_survival_time = np.exp(-lambda_R * Rx_speed * dt / Dt)
            for cluster in cluster_set_old:
                if np.random.rand() > p_survival_time:
                    cluster_set_old.remove(cluster)

            # 将两个簇集合合在一起，之后进行天线轴上的演进
            cluster_set = cluster_set_old + cluster_set_new

    # 3. 计算簇的小尺度衰落信道系数
    # LOS分量信道系数
    hL = generate_LOS_component(LOS_angle, Tx_ant, Rx_ant, simpar)

    # NLOS分量信道系数
    hN = generate_NLOS_component(cluster_set, Tx_ant, Rx_ant, simpar)

    # 4. 计算天线演进
    # TODO 完成天线轴演进，需要先确认场景
    Rx_cluster_set = cluster_evolution_Ant(cluster_number, Rx_ant)

    cluster_evolution_Ant_plot(Rx_cluster_set)

    # 三、按照场景合并信道系数，之后保存
    with open("channel_coeff.csv", "w", encoding="UTF-8",newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['index','coeff','delay'])
        if simpar.isLOS:
            pass
        else:
            for idx,ray in enumerate(hN):
                wr.writerow([idx,ray['coeff'],ray['delay']])
