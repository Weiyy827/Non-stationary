import numpy as np


class Cluster:
    """
    散射簇类

    成员：
    number: 子簇个数
    idx: 簇序号
    delay: 簇时延，单位s
    power: 簇功率
    cAOA,cZOA,cAOD,cZOD: 簇内子簇的到达角离开角，单位deg
    kappa: 子簇的极化交叉比，单位dB
    phase: 子簇的相位，单位rad
    """

    def __init__(self, delay, power, cAOA: list, cAOD: list, cZOA: list, cZOD: list, XPR, idx: int):
        """
        构造散射簇类
        :param delay: 簇时延，单位s
        :param power: 簇功率
        :param cAOA: 簇内子簇的AOA，单位deg
        :param cAOD: 簇内子簇的AOD，单位deg
        :param cZOA: 簇内子簇的ZOA，单位deg
        :param cZOD: 簇内子簇的ZOD，单位deg
        :param XPR: 簇极化交叉比，单位dB
        :param idx: 簇序号
        """
        # 簇参数设置
        self.number = 20
        self.idx = idx
        self.delay = delay
        self.power = power
        self.cAOA = cAOA
        self.cAOD = cAOD
        self.cZOA = cZOA
        self.cZOD = cZOD
        self.kappa = XPR + 13.65 * np.random.randn(20)
        self.phase = np.random.uniform(-np.pi, np.pi, [20, 4])


def generate_cluster(cluster_param):
    """
    生成簇内子簇的参数
    :param cluster_param: 包含时延，功率，到达离开角，XPR的簇参数
    :return: 包含簇对象的列表
    """
    tau, power, AOA, AOD, ZOA, ZOD, XPR = cluster_param
    cluster_set = []
    cDS, cASD, cASA, cZSA = 5, 3, 17, 7
    am = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
                   0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])
    # 生成簇内的角度
    for i in range(len(tau)):
        cAOA = AOA[i] + cASA * am
        cAOD = AOD[i] + cASD * am
        cZOA = ZOA[i] + cZSA * am
        cZOD = ZOD[i] + 3 / 8 * (10 ** 0.7) * am
        cluster_set.append(Cluster(tau[i], power[i], cAOA, cAOD, cZOA, cZOD, XPR, i))

    return cluster_set


def generate_cluster_param(lsp, cluster_num, LOS_angle):
    """
    生成簇参数
    :param lsp:当前时刻大尺度参数
    :param cluster_num: 簇数量
    :param LOS_angle: 当前时刻LOS径角度
    :return: 包括时延，功率，到达离开角，XPR的参数列表
    """
    LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD = LOS_angle
    cluster_num = int(cluster_num)

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

    return [tau, power, AOA, AOD, ZOA, ZOD, lsp['XPR']]
