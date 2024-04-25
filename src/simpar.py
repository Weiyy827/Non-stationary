import matplotlib.pyplot as plt


class Simulation_parameter:
    def __init__(self):
        self.sat = None
        self.fc = None
        self.rx = None
        self.tx = None
        self.snapshots = None

    def visualize_scenario(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.rx.position[0], self.rx.position[1], self.rx.position[2])
        ax.scatter(self.tx.position[0], self.tx.position[1], self.tx.position[2])
        plt.show()

import numpy as np
import scipy.constants

from src.config import Re, dt


class Antenna:
    """
    天线类

    成员：
    ant_type: 阵列类型
    num: 阵元数量
    position: 天线位置
    slant: 天线极化倾斜角
    azimuth: 天线阵面法向方位角
    elevation: 天线阵面法向仰角
    velocity: 天线速度
    """

    def __init__(self, position: list, angles: list, velocity: list, slant, **kwargs):

        if kwargs["Ant_type"] == "URA":
            self.ant_type = "URA"
            self.shape = kwargs["Shape"]
            self.num = self.shape[0] * self.shape[1]
        if kwargs["Ant_type"] == "ULA":
            self.ant_type = "ULA"
            self.ant_spacing = kwargs["Delta"]
            self.num = kwargs["Num"]

        self.position = np.array(position)
        self.slant = slant * np.pi / 180
        self.azimuth = angles[0] * np.pi / 180
        self.elevation = angles[1] * np.pi / 180
        self.velocity = np.array(velocity).reshape([3, ])

    def evolve(self):
        """
        天线在时间轴上演进一个单位
        :return: 更新后的天线位置
        """
        self.position += self.velocity * dt


class Satellite:
    def __init__(self, coord_type, height, azimuth, elevation):
        if coord_type == "lla":
            self.height = height
            self.azimuth = azimuth * np.pi / 180
            self.elevation = elevation * np.pi / 180
            earth_mass = 5.972e24
            self.vsat = np.sqrt(scipy.constants.gravitational_constant * earth_mass / (self.height + Re))
            self.velocity = np.array(
                [self.vsat * np.cos(self.azimuth), self.vsat * np.sin(self.azimuth), 0]
            ).reshape([1, 3])
            self.ecef_coordinate = np.array(
                [
                    (self.height + Re) * np.cos(self.elevation) * np.cos(self.azimuth),
                    (self.height + Re) * np.cos(self.elevation) * np.sin(self.azimuth),
                    (self.height + Re) * np.sin(self.elevation),
                ]
            )


class Origin:
    def __init__(self, elevation, azimuth):

        self.azimuth = azimuth * np.pi / 180
        self.elevation = elevation * np.pi / 180
        self.global_GCS_coordinate = np.array(
            [
                Re * np.cos(self.elevation) * np.cos(self.azimuth),
                Re * np.cos(self.elevation) * np.sin(self.azimuth),
                Re * np.sin(self.elevation),
            ]
        )

        z = (azimuth - 90) * np.pi / 180  # z：z轴不动逆时针旋转经度-90
        y = (90 - elevation) * np.pi / 180  # y: y轴不动顺时针时针旋转90-纬度
        self.rotation = np.array(
            [np.cos(z), np.sin(z), 0, -np.sin(z), np.cos(z), 0, 0, 0, 1]
        ).reshape([3, 3]) @ np.array(
            [np.cos(y), 0, -np.sin(y), 0, 1, 0, np.sin(y), 0, np.cos(y)]
        ).reshape(
            [3, 3]
        )
