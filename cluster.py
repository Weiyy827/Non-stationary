import numpy as np


class Cluster:
    def __init__(self, delay, power, cAOA, cAOD, cZOA, cZOD, XPR, idx: int):
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
