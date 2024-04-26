import numpy as np
from scipy.linalg import sqrtm


def generate_lsp_3GPP(Tx_ant, Rx_ant):
    """
    根据38811生成大尺度衰落参数

    :param Tx_ant: 发射天线对象
    :param Rx_ant: 接收天线对象
    :return: 包含大尺度衰落参数的字典，包括DS,KF,SF,ASD,ASA,ESD,ESA,XPR
    """
    vec = Tx_ant.position - Rx_ant.position
    d2D = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    alpha_rad = np.arctan(vec[2] / d2D)
    elevation = alpha_rad * 180 / np.pi
    parameters = get_lsp_parameter(elevation)

    cov = np.array([[0, -0.4, -0.4, 0.4, 0.8, -0.2, 0, 0],
                    [0, 0, 0, 0, -0.2, 0, 0, 0],
                    [0, 0, 0, -0.5, -0.5, 0, -0.8, 0],
                    [0, 0, 0, 0, 0, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, -0.3, 0.4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])
    cov = cov.T + cov + np.eye(8)

    mu = np.array([item[0] for item in parameters])
    sigma = np.array([item[1] for item in parameters])

    for i in range(8):
        cov[i, :] *= sigma[i]
        cov[:, i] *= sigma[i]

    lsp_3GPP = np.random.multivariate_normal(mu, cov)

    return {'DS': lsp_3GPP[0], 'KF': lsp_3GPP[1], 'SF': lsp_3GPP[2], 'ASD': lsp_3GPP[3], 'ASA': lsp_3GPP[4],
            'ESA': lsp_3GPP[5], 'ESD': lsp_3GPP[6], 'XPR': lsp_3GPP[6]}


def get_lsp_parameter(elevation):
    idx = int(np.round(elevation, decimals=-1) / 10 - 1)

    mu_DS = [-7.12, -7.28, -7.45, -7.73, -7.91, -8.14, -8.23, -8.28, -8.36][idx]
    sigma_DS = [0.80,0.67,0.68,0.66,0.62,0.51,0.45,0.31,0.08][idx]

    mu_KF = [4.4, 9, 9.3, 7.9, 7.4, 7, 6.9, 6.5, 6.8][idx]
    sigma_KF = [3.3, 6.6, 6.1, 4.0, 3, 2.6, 2.2, 2.1, 1.9][idx]

    mu_SF = 0
    sigma_SF = [3.5, 3.4, 2.9, 3, 3.1, 2.7, 2.5, 2.3, 1.2][idx]

    mu_ASD = [-3.06, -2.68, -2.51, -2.40, -2.31, -2.20, -2.00, -1.64, -0.63][idx]
    sigma_ASD = [0.48, 0.36, 0.38, 0.32, 0.33, 0.39, 0.40, 0.32, 0.53][idx]

    mu_ASA = [0.94, 0.87, 0.92, 0.79, 0.72, 0.60, 0.55, 0.71, 0.81][idx]
    sigma_ASA = [0.70, 0.66, 0.68, 0.64, 0.63, 0.54, 0.52, 0.53, 0.62][idx]

    mu_ZSA = [0.82, 0.50, 0.82, 1.23, 1.43, 1.56, 1.66, 1.73, 1.79][idx]
    sigma_ZSA = [0.03, 0.09, 0.05, 0.03, 0.06, 0.05, 0.05, 0.02, 0.01][idx]

    mu_ZSD = [-2.52, -2.29, -2.19, -2.24, -2.30, -2.48, -2.64, -2.68, -2.61][idx]
    sigma_ZSD = [0.50, 0.53, 0.58, 0.51, 0.46, 0.35, 0.31, 0.39, 0.28][idx]

    mu_XPR = [24.4, 23.6, 23.2, 22.6, 21.8, 20.5, 19.3, 17.4, 12.3][idx]
    sigma_XPR = [3.8, 4.7, 4.6, 4.9, 5.7, 6.9, 8.1, 10.3, 15.2][idx]

    return [[mu_DS, sigma_DS], [mu_KF, sigma_KF], [mu_SF, sigma_SF], [mu_ASD, sigma_ASD],
            [mu_ASA, sigma_ASA], [mu_ZSA, sigma_ZSA], [mu_ZSD, sigma_ZSD], [mu_XPR, sigma_XPR]]
