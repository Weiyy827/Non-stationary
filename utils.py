import numpy as np


def db2pow(db):
    return 10 ** (db / 10)


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


def calculate_LOS_angle(Tx_ant, Rx_ant):
    """
    计算LOS径的到达离开角
    :param Tx_ant: 发射天线对象
    :param Rx_ant: 接收天线对象
    :return: 包含LOS径AOA，AOD，ZOA，ZOD的列表，单位为deg
    """
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
    else:
        LOS_AOA = 2 * np.pi + np.arctan(vec[1] / vec[0]) / np.pi * 180
        LOS_AOD = LOS_AOA - 180

    # 转换为角度
    LOS_EOA = np.arctan(vec[2] / np.sqrt(vec[1] ** 2 + vec[0] ** 2)) / np.pi * 180
    LOS_ZOA = 90 - LOS_EOA
    LOS_ZOD = 180 - LOS_ZOA
    # 为函数传入参数
    LOS_Angle = [LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD]
    return LOS_Angle
