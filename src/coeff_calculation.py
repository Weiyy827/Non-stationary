import numpy as np
import scipy

from src.utils import db2pow, calc_delta_angel


def generate_LOS_component(LOS_angle, Tx_ant, Rx_ant, simpar):
    """计算LOS径的信道系数

    Args:
        LOS_angle (list[float]): LOS径的角度
        Tx_ant (Antenna): 发送天线
        Rx_ant (Antenna): 接收天线
        simpar (SimulationParameter): 仿真参数

    Returns:
        complex: LOS径的信道系数
    """
    LOS_AOA, LOS_AOD, LOS_ZOA, LOS_ZOD = LOS_angle

    # 计算LOS径与Rx天线法向的夹角
    LOS_Rx_delta_az, LOS_Rx_delta_ze = calc_delta_angel(LOS_AOA, LOS_ZOA, Rx_ant)

    # 方法相同计算Tx天线
    LOS_Tx_delta_az, LOS_Tx_delta_ze = calc_delta_angel(LOS_AOD, LOS_ZOD, Tx_ant)

    # 计算LOS径的三维方向矢量
    r_rx_LOS = np.array(
        [
            np.sin(LOS_ZOA * np.pi / 180) * np.cos(LOS_AOA * np.pi / 180),
            np.sin(LOS_ZOA * np.pi / 180) * np.sin(LOS_AOA * np.pi / 180),
            np.cos(LOS_ZOA * np.pi / 180),
        ]
    ).reshape([3, 1])

    r_tx_LOS = np.array(
        [
            np.sin(LOS_ZOD * np.pi / 180) * np.cos(LOS_AOD * np.pi / 180),
            np.sin(LOS_ZOD * np.pi / 180) * np.sin(LOS_AOD * np.pi / 180),
            np.cos(LOS_ZOD * np.pi / 180),
        ]
    ).reshape([3, 1])

    # 计算多普勒量
    vec = Rx_ant.position - Tx_ant.position
    doppler_1 = np.exp(
        -1j * 2 * np.pi * (np.sqrt(np.sum(vec**2))) / (scipy.constants.c / simpar.fc)
    )
    doppler_2 = np.exp(
        1j
        * 2
        * np.pi
        * (r_rx_LOS.T @ Rx_ant.position)
        / (scipy.constants.c / simpar.fc)
    )
    doppler_3 = np.exp(
        1j
        * 2
        * np.pi
        * (r_tx_LOS.T @ Tx_ant.position)
        / (scipy.constants.c / simpar.fc)
    )
    doppler_4 = np.exp(
        1j
        * 2
        * np.pi
        * (r_rx_LOS.T @ Rx_ant.velocity)
        / (scipy.constants.c / simpar.fc)
        * (simpar.snapshots - 1)
    )

    # 计算LOS径的信道系数,LOS径的时延设置为0
    hL = (
        np.transpose(
            calculate_Ant_pattern(LOS_Rx_delta_az, LOS_Rx_delta_ze, Rx_ant.slant)
        )
        @ np.array([1, 0, 0, -1]).reshape([2, 2])
        @ calculate_Ant_pattern(LOS_Tx_delta_az, LOS_Tx_delta_ze, Tx_ant.slant)
        * doppler_1
        * doppler_2
        * doppler_3
        * doppler_4
    )
    return hL


def generate_NLOS_component(cluster_set, Tx_ant, Rx_ant, simpar):
    """计算NLOS径信道系数

    Args:
        cluster_set (list[Cluster]): 散射簇集合
        Tx_ant (Antenna): 发送天线
        Rx_ant (Antenna): 接收天线
        simpar (SimulationParameter): 仿真参数

    Returns:
        list[dict{complex,float}]: NLOS径的信道系数与时延
    """
    fc = simpar.fc
    snapshots = simpar.snapshots
    hN = []

    # 对于集合中的第n个簇
    for cluster in cluster_set:

        # 对于簇内的第m个子簇
        for m in range(cluster.number):
            #  1.用子簇的AOA和ZOA计算接收场量
            ray_Rx_delta_az, ray_Rx_delta_ze = calc_delta_angel(
                cluster.ray_angle[m]["AOA"], cluster.ray_angle[m]["ZOA"], Rx_ant
            )
            Rx_field = calculate_Ant_pattern(
                ray_Rx_delta_az, ray_Rx_delta_ze, Rx_ant.slant
            )

            #  2.交叉极化项
            polar = cross_polar(cluster.kappa[m], cluster.phase[m])

            #  3.发射场量
            ray_Tx_delta_az, ray_Tx_delta_ze = calc_delta_angel(
                cluster.ray_angle[m]["AOD"], cluster.ray_angle[m]["ZOD"], Tx_ant
            )
            Tx_field = calculate_Ant_pattern(
                ray_Tx_delta_az, ray_Tx_delta_ze, Tx_ant.slant
            )

            #  4.多普勒量
            r_rx = np.array(
                [
                    np.sin(cluster.ray_angle[m]["ZOA"] * np.pi / 180)
                    * np.cos(cluster.ray_angle[m]["AOA"] * np.pi / 180),
                    np.sin(cluster.ray_angle[m]["ZOA"] * np.pi / 180)
                    * np.sin(cluster.ray_angle[m]["AOA"] * np.pi / 180),
                    np.cos(cluster.ray_angle[m]["ZOA"] * np.pi / 180),
                ]
            ).reshape([3, 1])
            r_tx = np.array(
                [
                    np.sin(cluster.ray_angle[m]["ZOD"] * np.pi / 180)
                    * np.cos(cluster.ray_angle[m]["AOD"] * np.pi / 180),
                    np.sin(cluster.ray_angle[m]["ZOD"] * np.pi / 180)
                    * np.sin(cluster.ray_angle[m]["AOD"] * np.pi / 180),
                    np.cos(cluster.ray_angle[m]["ZOD"] * np.pi / 180),
                ]
            ).reshape([3, 1])

            doppler_1 = np.exp(
                1j * 2 * np.pi * (r_rx.T @ Rx_ant.position) / (scipy.constants.c / fc)
            )
            doppler_2 = np.exp(
                1j * 2 * np.pi * (r_tx.T @ Tx_ant.position) / (scipy.constants.c / fc)
            )
            doppler_3 = np.exp(
                1j
                * 2
                * np.pi
                * (r_rx.T @ Rx_ant.velocity)
                / (scipy.constants.c / fc)
                * snapshots
            )

            # 子簇的信道系数之和为这个簇的信道系数
            coeff = (
                np.sqrt(cluster.ray_power[m])
                * np.transpose(Rx_field)
                @ polar
                @ Tx_field
                * doppler_1
                * doppler_2
                * doppler_3
            )[0][0]
        hN.append({"coeff": coeff, "delay": cluster.ray_delay[m]})
    return hN


def calculate_Ant_pattern(azimuth, zenith, slant):
    """计算天线的场量

    Args:
        azimuth (float): 径与天线法向的水平夹角，单位deg
        zenith (float): 径与天线法向的垂直夹角，单位deg
        slant (float): 天线的极化倾斜角，单位deg

    Returns:
        ndarray[[float],[float]]: 包含垂直场量和水平场量的2x1矩阵
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
    """计算子径的极化交叉量

    Args:
        xnm (float): 子径的极化交叉比，单位dB
        phase (list[float]): 子径的相位

    Returns:
        ndarray[[float,float],[float,float]]: 包含极化交叉量的2x2矩阵
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
