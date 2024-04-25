import numpy as np
from matplotlib import pyplot as plt


class Cluster:

    def __init__(self, lsp, cluster_param):
        # 簇参数设置
        number = 20
        cDS = 4.95e-9
        cASA = 3
        cESA = 1

        delta_angel = np.random.uniform(-2, 2, [number, 4])
        delta_tau = np.random.uniform(0, 2 * cDS, [number, ])
        self.ray_delay = cluster_param['tau'] + delta_tau - np.min(delta_tau)

        delta_power = (np.exp(-self.ray_delay / cDS) * np.exp(-np.sqrt(2) * np.abs(delta_angel[:, 0]) / cASA) * np.exp(
            -np.sqrt(2) * np.abs(delta_angel[:, 1]) / cASA) * np.exp(-np.sqrt(2) * np.abs(delta_angel[:, 2]) / cESA) *
                       np.exp(-np.sqrt(2) * np.abs(delta_angel[:, 3]) / cESA))

        self.ray_power = cluster_param['power'] * delta_power / np.sum(delta_power)
        self.kappa = lsp['XPR'] + 13.65 * np.random.randn(20)
        self.phase = np.random.uniform(-np.pi, np.pi, [20, 4])


def generate_cluster_param(lsp, cluster_num, LOS_angle):
    LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD = LOS_angle
    cluster_num = int(cluster_num)

    r_tau = 2.5
    DS = 10 ** lsp['DS']
    tau = -r_tau * DS * np.log(np.random.uniform(0, 1, [cluster_num, ]))
    # 将时延排序
    tau = np.sort(tau - np.min(tau))

    # 考虑LOS径存在的情况
    K = lsp['KF']
    c_tau = 0.7705 - 0.0433 * K + 0.0002 * K ** 2 + 0.000017 * K ** 3
    tau_LOS = tau / c_tau

    # 生成簇功率
    power = np.exp(-tau * (r_tau - 1) / (r_tau * DS)) * np.power(10, -3 * np.random.randn(cluster_num) / 10)
    power = power / np.sum(power)
    KR = 10 ** (K / 10)
    power_LOS = 1 / (KR + 1) * power
    power_LOS[0] += KR / (KR + 1)

    # 生成簇角度 参考38.901 Table 7.5-2
    ASA = 10 ** lsp['ASA']
    AOA = gen_cluster_azimuth(power_LOS, cluster_num, K, ASA, LOS_AOA)

    ASD = 10 ** lsp['ASD']
    AOD = gen_cluster_azimuth(power_LOS, cluster_num, K, ASD, LOS_AOD)

    ZSA = 10 ** lsp['ESA']
    ZOA = gen_cluster_zenith(power_LOS, cluster_num, K, ZSA, LOS_ZOA)

    ZSD = 10 ** lsp['ESD']
    ZOD = gen_cluster_zenith(power_LOS, cluster_num, K, ZSD, LOS_ZOD)

    cluster_param = []
    for i in range(cluster_num):
        cluster_param.append(
            {'tau': tau[i], 'power': power[i], 'AOA': AOA[i], 'AOD': AOD[i], 'ZOA': ZOA[i], 'ZOD': ZOD[i]})
    return cluster_param


def gen_cluster_azimuth(power_LOS, cluster_num, K, AS, LOS_az):
    c_phi = 0.779 * (1.1035 - 0.028 * K - 0.002 * K ** 2 + 0.0001 * K ** 3)

    AZ_temp = 2 * (AS / 1.4) * np.sqrt(-np.log(power_LOS / np.max(power_LOS))) / c_phi

    Xn = np.random.choice([-1, 1], cluster_num)
    Yn = AS / 7 * np.random.randn(cluster_num)

    AZ = (Xn * AZ_temp + Yn) - (Xn[0] * AZ_temp[0] + Yn[0] - LOS_az)

    return np.mod(AZ, 360)


def gen_cluster_zenith(power_LOS, cluster_num, K, AS, LOS_ze):
    c_theta = 0.889 * (1.3086 + 0.0339 * K - 0.0077 * K ** 2 + 0.0002 * K ** 3)

    ZE_temp = - AS * np.log(power_LOS / np.max(power_LOS)) / c_theta

    Xn = np.random.choice([-1, 1], cluster_num)
    Yn = AS / 7 * np.random.randn(cluster_num)

    ZE = (Xn * ZE_temp + Yn) - (Xn[0] * ZE_temp[0] + Yn[0] - LOS_ze)

    return np.mod(np.abs(ZE), 360)


def plot_PDP(cluster_set):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, cluster in enumerate(cluster_set):
        plt.stem(cluster.ray_delay * 1e6, 10 * np.log10(cluster.ray_power), linefmt=colors[idx], markerfmt=colors[idx],
                 bottom=-50)

    plt.xlabel('Ray Delay [us]')
    plt.ylabel('Power [dBW]')
    plt.title('Power-Delay Profile')
    plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
    plt.show()
