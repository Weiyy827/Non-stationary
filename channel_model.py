import numpy as np
import scipy

import LSP
import scenario
import config
from cluster import cluster_generate
from evolution import cluster_evolution_Ant_plot, cluster_evolution_Ant
import coeff_calculation


def non_stationary_channel(
        inputWaveform, Tx_ant: scenario.Antenna, Rx_ant: scenario.Antenna, fc, bw, time_instance
):
    # 0.确定LOS径的到达角和离开角。方位角(0,2pi),天顶角(0，pi)
    vec = Tx_ant.position - Rx_ant.position
    if vec[0] > 0 and vec[1] > 0:
        LOS_AOA = np.arctan(vec[1] / vec[0]) / np.pi * 180
        LOS_AOD = 180 + LOS_AOA
    elif vec[0] < 0 < vec[1]:
        LOS_AOA = np.pi + np.arctan(vec[1] / vec[0]) / np.pi * 180
        LOS_AOD = 180 + LOS_AOA
    elif vec[0] < 0 and vec[1] < 0:
        LOS_AOA = np.pi + np.arctan(vec[1] / vec[0]) / np.pi * 180
        LOS_AOD = LOS_AOA - 180
    elif vec[0] > 0 > vec[1]:
        LOS_AOA = 2 * np.pi + np.arctan(vec[1] / vec[0]) / np.pi * 180
        LOS_AOD = LOS_AOA - 180
    # 转换为角度
    LOS_EOA = np.arctan(vec[2] / np.sqrt(vec[1] ** 2 + vec[0] ** 2)) / np.pi * 180
    LOS_ZOA = 90 - LOS_EOA
    LOS_ZOD = 180 - LOS_ZOA

    # 1.计算该时刻大尺度衰落参数,单位log10(s),dB,log10(deg)，参考quadriga
    Tx_ant.evolve(time_instance)
    Rx_ant.evolve(time_instance)
    lsp = LSP.lsp_generate(Tx_ant, Rx_ant, fc)

    # 2.生成簇的参数，参考38.901
    cluster_num = 20
    # 生成时延
    r_tau = 3
    DS = 10 ** lsp['DS']
    tau = -r_tau * DS * np.log(np.random.uniform(0, 1, [cluster_num, ]))
    tau = np.sort(tau - np.min(tau))

    K = lsp['KF']
    c_tau = 0.7705 - 0.0433 * K + 0.0002 * K ** 2 + 0.000017 * K ** 3
    tau /= c_tau

    # 生成功率
    power = np.exp(-tau * (r_tau - 1) / (r_tau * DS)) * np.power(10, -3 * np.random.randn(cluster_num) / 10)
    # 增加LOS径的功率
    KR = 10 ** (K / 10)
    power = power / np.sum(power) / (KR + 1)

    # 生成角度 参考38.901 Table 7.5-2
    ASA = 10 ** lsp['ASA']
    c_phi = 1.289 * (1.1035 - 0.028 * K - 0.002 * K ** 2 + 0.0001 * K ** 3)
    AOA = 2 * (ASA / 1.4) * np.sqrt(-np.log(power / np.max(power))) / c_phi
    AOA = ((np.random.uniform(-1, 1, cluster_num) * AOA + (ASA / 7) * np.random.randn(cluster_num))
           - (np.random.uniform(-1, 1) * AOA[0] + np.random.randn() - LOS_AOA))

    ASD = 10 ** lsp['ASD']
    AOD = 2 * (ASD / 1.4) * np.sqrt(-np.log(power / np.max(power))) / c_phi
    AOD = ((np.random.uniform(-1, 1, cluster_num) * AOD + (ASD / 7) * np.random.randn(cluster_num))
           - (np.random.uniform(-1, 1) * AOD[0] + np.random.randn() - LOS_AOD))

    ZSA = 10 ** lsp['ESA']
    c_theta = 1.178 * (1.3086 - 0.0339 * K - 0.0077 * K ** 2 + 0.0002 * K ** 3)
    ZOA = - ZSA * np.log(power / np.max(power)) / c_theta
    ZOA = ((np.random.uniform(-1, 1, cluster_num) * ZOA + (ZSA / 7) * np.random.randn(cluster_num))
           - (np.random.uniform(-1, 1) * ZOA[0] + np.random.randn() - LOS_ZOA))

    ZSD = 10 ** lsp['ESD']
    ZOD = - ZSD * np.log(power / np.max(power)) / c_theta
    ZOD = ((np.random.uniform(-1, 1, cluster_num) * ZOD + (ZSD / 7) * np.random.randn(cluster_num))
           - (np.random.uniform(-1, 1) * ZOD[0] + np.random.randn() - LOS_ZOA))

    # 用参数生成簇
    cluster_set = cluster_generate(tau, power, AOA, AOD, ZOA, ZOD, lsp['XPR'])

    # 2.计算该时刻的簇演进
    # 在天线轴上进行演进
    Rx_cluster_set = cluster_evolution_Ant(cluster_set, Rx_ant)
    cluster_evolution_Ant_plot(Rx_cluster_set)

    # 该时刻下的信道系数
    # 计算小尺度衰落信道系数
    h = np.zeros([Rx_ant.num, Tx_ant.num], complex)

    faraday = np.array([np.cos(108 / (fc ** 2)), -np.sin(108 / (fc ** 2)),
                        np.sin(108 / (fc ** 2)), np.cos(108 / (fc ** 2))]).reshape([2, 2])

    # 计算场量
    # 将角度转换到天线为原点的坐标系上

    r_rx_LOS = np.array(
        [np.sin(LOS_ZOA) * np.cos(LOS_AOA), np.sin(LOS_ZOA) * np.sin(LOS_AOA),
         np.cos(LOS_ZOA)]).reshape([3, 1])
    r_tx_LOS = np.array(
        [np.sin(LOS_ZOD) * np.cos(LOS_AOD), np.sin(LOS_ZOD) * np.sin(LOS_AOD),
         np.cos(LOS_ZOD)]).reshape([3, 1])
    doppler_1 = np.exp(-1j * 2 * np.pi * (np.sqrt(np.sum(vec ** 2))) / (scipy.constants.c / fc))
    doppler_2 = np.exp(1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.position) / (scipy.constants.c / fc))
    doppler_3 = np.exp(1j * 2 * np.pi * (r_tx_LOS.T @ Tx_ant.position) / (scipy.constants.c / fc))
    doppler_4 = np.exp(
        1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * time_instance)

    hL = (
            np.transpose(
                coeff_calculation.field(LOS_ZOA, LOS_AOA, Rx_ant.slant)
            )
            @ np.array([1, 0, 0, -1]).reshape([2, 2])
            @ coeff_calculation.field(LOS_ZOD, LOS_AOD, Tx_ant.slant)
            * doppler_1
            * doppler_2
            * doppler_3
            * doppler_4
    )
    # NLOS分量

    for i in range(Rx_ant.num):  # 第i个Rx天线对所有Tx天线
        for j in range(Tx_ant.num):  # 第i个Rx天线对第j个Tx天线
            clusters = Rx_cluster_set[i]  # 第i个Rx天线上的可见簇集合
            hN = 0
            for ray in clusters:  # 第k个可见簇
                hN_k = 0
                for m in range(ray.number):  # 簇内的子簇
                    #  1.用子簇的Rx A和E计算接收场量
                    Rx_field = coeff_calculation.field(
                        ray.cZOA[m], ray.cAOA[m], Rx_ant.slant
                    )

                    #  2.交叉极化项
                    cross_polar = coeff_calculation.cross_polar(
                        ray.kappa[m], ray.phase[m]
                    )

                    #  3.法拉第旋转，在载波频率小于10GHz时需要考虑

                    #  4.发射场量
                    Tx_field = coeff_calculation.field(
                        ray.cZOD[m], ray.cAOD[m], Tx_ant.slant
                    )

                    #  5.多普勒量
                    r_rx = np.array(
                        [np.sin(ray.cZOA[m]) * np.cos(ray.cAOA[m]), np.sin(ray.cZOA[m]) * np.sin(ray.cAOA[m]),
                         np.cos(ray.cZOA[m])]).reshape([3, 1])
                    r_tx = np.array(
                        [np.sin(ray.cZOD[m]) * np.cos(ray.cAOD[m]), np.sin(ray.cZOD[m]) * np.sin(ray.cAOD[m]),
                         np.cos(ray.cZOD[m])]).reshape([3, 1])

                    doppler_1 = np.exp(1j * 2 * np.pi * (r_rx.T @ Rx_ant.position) / (scipy.constants.c / fc))
                    doppler_2 = np.exp(1j * 2 * np.pi * (r_tx.T @ Tx_ant.position) / (scipy.constants.c / fc))
                    doppler_3 = np.exp(
                        1j * 2 * np.pi * (r_rx.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * time_instance)

                    #  6.时延项
                    #  时延滤波器，将时间时延转换为采样时延,采样间隔1ns
                    hN_k += (
                            np.sqrt(ray.power / ray.number)
                            * np.transpose(Rx_field)
                            @ cross_polar
                            @ faraday
                            @ Tx_field
                            * doppler_1
                            * doppler_2
                            * doppler_3
                    )
                hN += hN_k
            h[i][j] = np.sqrt(KR / (KR + 1)) * hL + np.sqrt(1 / (KR + 1)) * hN
    return h
