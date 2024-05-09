import numpy as np

from src.config import Re
from src.simpar import Antenna


def db2pow(db) -> float:
    """
    将功率的dB值转换为线性值
    :param db: 功率dB值
    :return: 功率线性值
    """
    return 10 ** (db / 10)


def calc_delta_angel(azimuth, zenith, ant: Antenna) -> tuple[float, float]:
    """
    计算传播路径与天线法向的垂直夹角与水平夹角

    :param azimuth: 传播路径在GCS坐标系下的方位角，单位deg
    :param zenith: 传播路径在GCS坐标系下的天顶角，单位deg
    :param ant: 天线
    :return: delta_az, delta_ze:与天线法向的水平夹角和垂直夹角，单位deg
    """
    # 计算向量
    vec = np.array([np.sin(zenith / 180 * np.pi) * np.cos(azimuth / 180 * np.pi),
                    np.sin(zenith / 180 * np.pi) * np.sin(azimuth / 180 * np.pi),
                    np.cos(zenith / 180 * np.pi)
                    ])

    # 旋转x轴和z轴，其中x轴为天线平面法向
    xyz = np.eye(3)

    rotate_z = np.array([[np.cos(ant.azimuth), -np.sin(ant.azimuth), 0],
                         [np.sin(ant.azimuth), np.cos(ant.azimuth), 0],
                         [0, 0, 1]])
    rotate_y = np.array([[np.cos(ant.elevation), 0, -np.sin(ant.elevation)],
                         [0, 1, 0],
                         [np.sin(ant.elevation), 0, np.cos(ant.elevation)]])

    xyz_rotated = rotate_z @ rotate_y @ xyz
    normal = xyz_rotated[:, 2]
    # 计算方向向量和x轴的垂直夹角和水平夹角
    delta_ze = np.abs(
        90 - np.arccos(np.dot(vec, normal) / (np.linalg.norm(vec) * np.linalg.norm(normal))) / np.pi * 180)
    vec_projection = vec - np.dot(np.dot(vec, normal), normal)
    delta_az = np.arccos(np.dot(vec_projection, xyz_rotated[:, 0]) / (
            np.linalg.norm(vec_projection) * np.linalg.norm(xyz_rotated[:, 0]))) / np.pi * 180

    return delta_az, delta_ze


def ecef2gcs(ecef, original) -> list[float]:
    """
    将ecef坐标转换到GCS坐标

    :param ecef: ecef坐标系下的坐标，单位m
    :param original: GCS坐标系原点在ecef坐标系下的方位角和仰角，单位deg
    :return:
    """
    azimuth, elevation = original
    z = (azimuth - 90) * np.pi / 180  # z：z轴不动逆时针旋转经度-90
    y = (90 - elevation) * np.pi / 180  # y: y轴不动顺时针时针旋转90-纬度
    azimuth_rad = azimuth / 180 * np.pi
    elevation_rad = elevation / 180 * np.pi

    rotation = np.array(
        [np.cos(z), np.sin(z), 0, -np.sin(z), np.cos(z), 0, 0, 0, 1]
    ).reshape([3, 3]) @ np.array(
        [np.cos(y), 0, -np.sin(y), 0, 1, 0, np.sin(y), 0, np.cos(y)]
    ).reshape(
        [3, 3]
    )

    ecef_original = np.array(
        [
            Re * np.cos(elevation_rad) * np.cos(azimuth_rad),
            Re * np.cos(elevation_rad) * np.sin(azimuth_rad),
            Re * np.sin(elevation_rad),
        ]
    )

    return (rotation @ (ecef.reshape([3, 1]) - ecef_original.reshape([3, 1]))).reshape([3, ])


def calculate_LOS_angle(Tx_ant, Rx_ant) -> list[float]:
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
