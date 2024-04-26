import numpy as np
import scipy

from src.utils import db2pow, gcs2lcs


def generate_LOS_component(LOS_angle, Tx_ant, Rx_ant, simpar):
    LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD = LOS_angle
    # 将角度转换到天线为原点的坐标系上
    LOS_Rx_delta_az, LOS_Rx_delta_ze = gcs2lcs(LOS_AOA, LOS_ZOA, Rx_ant)

    # 方法相同计算Tx天线为原点的LCS坐标
    LOS_Tx_delta_az, LOS_Tx_delta_ze = gcs2lcs(LOS_AOD, LOS_ZOD, Tx_ant)

    # 计算LOS径的三维方向矢量
    r_rx_LOS = np.array(
        [np.sin(LOS_ZOA * np.pi / 180) * np.cos(LOS_AOA * np.pi / 180),
         np.sin(LOS_ZOA * np.pi / 180) * np.sin(LOS_AOA * np.pi / 180),
         np.cos(LOS_ZOA * np.pi / 180)])

    r_tx_LOS = np.array(
        [np.sin(LOS_ZOD * np.pi / 180) * np.cos(LOS_AOD * np.pi / 180),
         np.sin(LOS_ZOD * np.pi / 180) * np.sin(LOS_AOD * np.pi / 180),
         np.cos(LOS_ZOD * np.pi / 180)])

    # 计算多普勒量
    vec = Rx_ant.position - Tx_ant.position
    doppler_1 = np.exp(-1j * 2 * np.pi * (np.sqrt(np.sum(vec ** 2))) / (scipy.constants.c / simpar.fc))
    doppler_2 = np.exp(1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.position) / (scipy.constants.c / simpar.fc))
    doppler_3 = np.exp(1j * 2 * np.pi * (r_tx_LOS.T @ Tx_ant.position) / (scipy.constants.c / simpar.fc))
    doppler_4 = np.exp(
        1j * 2 * np.pi * (r_rx_LOS.T @ Rx_ant.velocity) / (scipy.constants.c / simpar.fc) * (simpar.snapshots - 1))

    # 计算LOS径的信道系数,LOS径的时延设置为0
    hL = (
            np.transpose(calculate_Ant_pattern(LOS_Rx_delta_az, LOS_Rx_delta_ze, Rx_ant.slant))
            @ np.array([1, 0, 0, -1]).reshape([2, 2])
            @ calculate_Ant_pattern(LOS_Tx_delta_az, LOS_Tx_delta_ze, Tx_ant.slant)
            * doppler_1
            * doppler_2
            * doppler_3
            * doppler_4
    )
    return hL


def calculate_Ant_pattern(zenith, azimuth, slant):
    """
    计算天线的场量

    :param zenith: LCS坐标系下径的天顶角，单位deg
    :param azimuth: LCS坐标系下径的方位角，单位deg
    :param slant: 天线的极化倾斜角，单位deg
    :return: 包含垂直场量和水平场量的2x1矩阵
    """
    vertical_cut = -np.min([12 * (zenith / 65) ** 2, 30])
    horizontal_cut = -np.min([12 * (azimuth / 65) ** 2, 30])
    radiation_field = -np.min([-vertical_cut - horizontal_cut, 30])
    power_pattern = np.array(
        [
            np.sqrt(db2pow(radiation_field)) * np.cos(slant),
            np.sqrt(db2pow(radiation_field)) * np.sin(slant),
        ]
    ).reshape([2, 1])
    return power_pattern


def cross_polar(xnm, phase):
    """
    计算子径的极化交叉量

    :param xnm: 子径的极化交叉比，单位dB
    :param phase: 子径的相位
    :return: 包含极化交叉量的2x2矩阵
    """
    xpr = 10 ** (xnm / 10)
    result = np.array(
        [
            np.exp(1j * phase[0]),
            np.sqrt(xpr ** -1) * np.exp(1j * phase[1]),
            np.sqrt(xpr ** -1) * np.exp(1j * phase[2]),
            np.exp(1j * phase[3]),
        ]
    ).reshape([2, 2])
    return result