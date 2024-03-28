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


def cluster_generate(tau, power, AOA, AOD, ZOA, ZOD, XPR):
    """
    生成簇集合

    :param tau: 簇时延，单位s
    :param power: 簇功率
    :param AOA: 簇AOA，单位deg
    :param AOD: 簇AOD，单位deg
    :param ZOA: 簇ZOA，单位deg
    :param ZOD: 簇ZOD，单位deg
    :param XPR: 簇极化交叉比，单位dB
    :return: 包含指定数量个数的簇对象的列表
    """
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