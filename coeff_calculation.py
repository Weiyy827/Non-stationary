import numpy as np

from utils import db2pow


def field(zenith, azimuth, slant):
    """
    计算天线的场量

    :param zenith: LCS坐标系下径的天顶角，单位deg
    :param azimuth: LCS坐标系下径的方位角，单位deg
    :param slant: 天线的极化倾斜角，单位deg
    :return: 包含垂直场量和水平场量的2x1矩阵
    """
    slant = slant / 180 * np.pi
    vertical_cut = -np.min([12 * ((zenith - 90) / 65) ** 2, 30])
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
            np.sqrt(xpr**-1) * np.exp(1j * phase[1]),
            np.sqrt(xpr**-1) * np.exp(1j * phase[2]),
            np.exp(1j * phase[3]),
        ]
    ).reshape([2, 2])
    return result
