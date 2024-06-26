import numpy as np
from scipy.linalg import sqrtm


def generate_lsp_quadriga(Tx_ant, Rx_ant, fc):
    """
    根据Quadriga_Documentary中的方式生成大尺度衰落参数

    :param Tx_ant: 发射天线对象
    :param Rx_ant: 接收天线对象
    :param fc: 载波频率
    :return: 包含大尺度衰落参数的字典，包括DS,KF,SF,ASD,ASA,ESD,ESA,XPR
    """

    # 1.计算rx和tx之间的参数
    vec = Tx_ant.position - Rx_ant.position
    d2D = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    alpha_rad = np.arctan(vec[2] / d2D)
    fc /= 1e9

    # 2.
    X = np.random.randn(8, 1)
    inter_corr = np.array([[1, -0.8, 0.2, 0.8, 0.8, 0.8, 0.8, 0],
                           [-0.8, 1, -0.3, -0.8, -0.8, -0.8, -0.8, 0],
                           [0.2, -0.3, 1, 0.2, 0.2, 0.2, 0.2, 0],
                           [0.8, -0.8, 0.2, 1, 0.8, 0.8, 0.8, 0],
                           [0.8, -0.8, 0.2, 0.8, 1, 0.8, 0.8, 0],
                           [0.8, -0.8, 0.2, 0.8, 0.8, 1, 0.8, 0],
                           [0.8, -0.8, 0.2, 0.8, 0.8, 0.8, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])
    X = sqrtm(inter_corr) @ X
    DS = -7.95 - 0.4 * np.log10(fc) + 0.4 * np.log10(alpha_rad) + X[0][0] * 0.7
    KF = 22.45 + 7.9 * np.log10(fc) - 10.95 * np.log10(alpha_rad) + X[1][0] * (
            10.65 + 2.2 * np.log10(fc) - 2.65 * np.log10(alpha_rad))
    SF = X[2][0] * (0.15 - 0.6 * np.log10(alpha_rad))

    ASD = 1.85 - 0.4 * np.log10(fc) - 1 * np.log10(d2D) + 0.3 * np.log10(alpha_rad) + X[3][0] * 0.7
    ASD = np.min([ASD, np.log10(104)])

    ASA = 0.9 - 0.4 * np.log10(fc) + 0.55 * np.log10(alpha_rad) + X[4][0] * 0.7
    ASA = np.min([ASA, np.log10(104)])

    ESD = 1.75 - 0.4 * np.log10(fc) - 1 * np.log10(d2D) + 0.5 * np.log10(alpha_rad) + X[5][0] * 0.7
    ESD = np.min([ESD, np.log10(52)])

    ESA = 0.3 - 0.4 * np.log10(fc) + 0.7 * np.log10(alpha_rad) + X[6][0] * 0.7
    ESA = np.min([ESA, np.log10(52)])

    XPR = 15.15 - 13.45 * np.log10(alpha_rad) + X[7][0] * (13.65 + 8.85 * np.log10(alpha_rad))

    return {'DS': DS, 'KF': KF, 'SF': SF, 'ASD': ASD, 'ASA': ASA, 'ESD': ESD, 'ESA': ESA, 'XPR': XPR}
