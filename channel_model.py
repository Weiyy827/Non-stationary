import matplotlib.pyplot as plt
import numpy as np
import scipy

import LSP
from coeff_calculation import field,cross_polar
import scenario
from cluster import cluster_generate
from config import fs
from evolution import cluster_evolution_Ant_plot, cluster_evolution_Ant


def LCS_convert(azimuth, zenith, ant):
    """
    将GCS下的方位角和天顶角转换到对应天线的LCS坐标中

    :param azimuth: GCS下的方位角，单位deg
    :param zenith: GCS下的天顶角，单位deg
    :param ant: LCS的原点天线
    :return: LCS下的方位角和天顶角，单位deg
    """
    #  计算单位坐标
    coordinate = np.array([np.cos(zenith / 180 * np.pi) * np.cos(azimuth / 180 * np.pi),
                           np.cos(zenith / 180 * np.pi) * np.sin(azimuth / 180 * np.pi),
                           np.sin(zenith / 180 * np.pi)
                           ]).reshape([3, 1])
    # 把这一点变换到LCS上:1. 绕y轴顺时针旋转90度;2. 绕x轴顺时针转ant_azimuth度；3.绕y轴逆时针转ant_elevation度
    rotation1 = np.array([np.cos(np.pi / 2), 0, -np.sin(np.pi / 2),
                          0, 1, 0,
                          np.sin(np.pi / 2), 0, np.cos(np.pi / 2)
                          ]).reshape([3, 3])
    rotation2 = np.array([1, 0, 0,
                          0, np.cos(ant.azimuth), np.sin(ant.azimuth),
                          0, -np.sin(ant.azimuth), np.cos(ant.azimuth)
                          ]).reshape([3, 3])
    rotation3 = np.array([np.cos(ant.elevation), 0, np.sin(ant.elevation),
                          0, 1, 0,
                          -np.sin(ant.elevation), 0, np.cos(ant.elevation)
                          ]).reshape([3, 3])
    LCS_coordinate = rotation1 @ rotation2 @ rotation3 @ coordinate
    # 计算LCS下的A和Z
    if LCS_coordinate[0] > 0 and LCS_coordinate[1] > 0:
        azimuth_LCS = np.arctan(LCS_coordinate[1] / LCS_coordinate[0]) / np.pi * 180
    elif LCS_coordinate[0] < 0 < LCS_coordinate[1]:
        azimuth_LCS = np.pi + np.arctan(LCS_coordinate[1] / LCS_coordinate[0]) / np.pi * 180
    elif LCS_coordinate[0] < 0 and LCS_coordinate[1] < 0:
        azimuth_LCS = np.pi + np.arctan(LCS_coordinate[1] / LCS_coordinate[0]) / np.pi * 180
    elif LCS_coordinate[0] > 0 > LCS_coordinate[1]:
        azimuth_LCS = 2 * np.pi + np.arctan(LCS_coordinate[1] / LCS_coordinate[0]) / np.pi * 180

    # 转换为角度
    elevation_LCS = np.arctan(
        LCS_coordinate[2] / np.sqrt(LCS_coordinate[1] ** 2 + LCS_coordinate[0] ** 2)) / np.pi * 180
    zenith_LCS = 90 - elevation_LCS

    return azimuth_LCS[0], zenith_LCS[0]


def non_stationary_channel(
        inputWaveform, Tx_ant: scenario.Antenna, Rx_ant: scenario.Antenna, fc, time_instance
):
    """
    将波形过信道

    :param inputWaveform: 输入波形
    :param Tx_ant: 发送端天线对象
    :param Rx_ant: 接收端天线
    :param fc: 载波频率，单位为Hz
    :param time_instance: 时刻单位
    :return: 过信道后的波形
    """

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

    # 天线对象时间演进
    Tx_ant.evolve(time_instance)
    Rx_ant.evolve(time_instance)
    lsp = LSP.lsp_generate(Tx_ant, Rx_ant, fc)

    # 2.生成所有簇的参数，参考38.901
    cluster_num = 20

    # 生成簇时延
    r_tau = 3
    DS = 10 ** lsp['DS']
    tau = -r_tau * DS * np.log(np.random.uniform(0, 1, [cluster_num, ]))
    # 将时延排序
    tau = np.sort(tau - np.min(tau))

    # 考虑LOS径存在的情况
    K = lsp['KF']
    c_tau = 0.7705 - 0.0433 * K + 0.0002 * K ** 2 + 0.000017 * K ** 3
    tau /= c_tau

    # 生成簇功率
    power = np.exp(-tau * (r_tau - 1) / (r_tau * DS)) * np.power(10, -3 * np.random.randn(cluster_num) / 10)
    # 将NLOS径的功率除以KR+1
    KR = 10 ** (K / 10)
    power = power / np.sum(power) / (KR + 1)

    # 生成簇角度 参考38.901 Table 7.5-2
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

    # 3. 用参数生成簇
    cluster_set = cluster_generate(tau, power, AOA, AOD, ZOA, ZOD, lsp['XPR'])

    # 画出簇的时延功率谱
    delays = [i.delay for i in cluster_set]
    powers = [i.power for i in cluster_set]
    plt.scatter(delays, powers)
    plt.xlabel("Delay")
    plt.ylabel("Power")
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
    # 将角度转换到天线为原点的坐标系上
    LOS_AOA_LCS, LOS_ZOA_LCS = LCS_convert(LOS_AOA, LOS_ZOA, Rx_ant)

    # 方法相同计算Tx天线为原点的LCS坐标
    LOS_AOD_LCS, LOS_ZOD_LCS = LCS_convert(LOS_AOD, LOS_ZOD, Tx_ant)

    # 计算LOS径的三维方向矢量
    r_rx_LOS = np.array(
        [np.sin(LOS_ZOA) * np.cos(LOS_AOA), np.sin(LOS_ZOA) * np.sin(LOS_AOA),
         np.cos(LOS_ZOA)]).reshape([3, 1])

    r_tx_LOS = np.array(
        [np.sin(LOS_ZOD) * np.cos(LOS_AOD), np.sin(LOS_ZOD) * np.sin(LOS_AOD),
         np.cos(LOS_ZOD)]).reshape([3, 1])

    # 计算多普勒量
    doppler_1 = np.exp(-1j * 2 * np.pi * (np.sqrt(np.sum(vec ** 2))) / (scipy.constants.c / fc))
    doppler_2 = np.exp(1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.position) / (scipy.constants.c / fc))
    doppler_3 = np.exp(1j * 2 * np.pi * (r_tx_LOS.T @ Tx_ant.position) / (scipy.constants.c / fc))
    doppler_4 = np.exp(
        1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * time_instance)

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
    # 储存输出波形
    outputWaveform = np.zeros([1, len(inputWaveform)], dtype=complex)

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
                    ray_cAOA_LCS, ray_cZOA_LCS = LCS_convert(cluster.cAOA[m], cluster.cZOA[m], Rx_ant)
                    Rx_field = field(ray_cZOA_LCS, ray_cAOA_LCS, Rx_ant.slant)

                    #  2.交叉极化项
                    polar = cross_polar(cluster.kappa[m], cluster.phase[m])

                    #  3.法拉第旋转，在载波频率小于10GHz时需要考虑
                    faraday = np.array([np.cos(108 / (fc ** 2)), -np.sin(108 / (fc ** 2)),
                                        np.sin(108 / (fc ** 2)), np.cos(108 / (fc ** 2))]).reshape([2, 2])

                    #  4.发射场量
                    ray_cAOD_LCS, ray_cZOD_LCS = LCS_convert(cluster.cAOD[m], cluster.cZOD[m], Rx_ant)
                    Tx_field = field(ray_cZOD_LCS, ray_cAOD_LCS, Tx_ant.slant)

                    #  5.多普勒量
                    r_rx = np.array(
                        [np.sin(cluster.cZOA[m]) * np.cos(cluster.cAOA[m]), np.sin(cluster.cZOA[m]) * np.sin(cluster.cAOA[m]),
                         np.cos(cluster.cZOA[m])]).reshape([3, 1])
                    r_tx = np.array(
                        [np.sin(cluster.cZOD[m]) * np.cos(cluster.cAOD[m]), np.sin(cluster.cZOD[m]) * np.sin(cluster.cAOD[m]),
                         np.cos(cluster.cZOD[m])]).reshape([3, 1])

                    doppler_1 = np.exp(1j * 2 * np.pi * (r_rx.T @ Rx_ant.position) / (scipy.constants.c / fc))
                    doppler_2 = np.exp(1j * 2 * np.pi * (r_tx.T @ Tx_ant.position) / (scipy.constants.c / fc))
                    doppler_3 = np.exp(
                        1j * 2 * np.pi * (r_rx.T @ Rx_ant.velocity) / (scipy.constants.c / fc) * time_instance)

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

                # 这个簇的时延，转化成采样时延
                delay = cluster.delay * fs

                # 计算时延滤波器系数
                filter_coeff = np.append(np.zeros(int(delay)), 1)

                # 对信号进行时延处理，之后累加
                outputWaveform += hN * scipy.signal.lfilter(filter_coeff, 1, inputWaveform)

    # 计算完32x32个NLOS分量后，为信号添加LOS分量
    outputWaveform += np.sqrt(KR / (KR + 1)) * hL * inputWaveform

    # 返回输出波形
    return outputWaveform
