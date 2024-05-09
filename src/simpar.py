import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

from src.config import Re, dt


class SimulationParameter:
    """
    用来储存和传递仿真参数的类

    属性：
    sat:Satellite
        仿真场景卫星对象
    fc:float
        载波频率，单位Hz
    tx,rx:Antenna
        仿真场景发射天线和接收天线对象
    snapshots:int
        仿真时长，单位为时刻数，每个时间间隔为一个平稳时间
    original:list[float]
        仿真场景原点在地球表面上的坐标，采用lla坐标系，单位deg

    方法:
    visualize_scenario() -> None
        显示接收端和发射端的位置场景图

    """

    def __init__(self):
        self.sat = None
        self.fc = None
        self.rx = None
        self.tx = None
        self.snapshots = None
        self.original = None
        self.isLOS = None

    def visualize_scenario(self):
        """
        显示接收端和发射端的位置场景图
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.rx.position[0], self.rx.position[1], self.rx.position[2])
        ax.scatter(self.tx.position[0], self.tx.position[1], self.tx.position[2])
        plt.show()


class Antenna:
    """
    储存天线属性的类

    属性：
    ant_type:str
        阵列类型，可选择URA与ULA（目前仅支持ULA）
    num:int
        阵元数量
    position:list[float]
        天线在GCS坐标系中的位置，单位m
    delta:float
        阵元间距，单位m
    slant:float
        天线极化倾斜角，单位deg
    azimuth:float
        天线阵面法向方位角，单位deg
    elevation:float
        天线阵面法向仰角，单位deg
    velocity:list[float]
        天线速度矢量，单位m/s

    方法：
    evolve() -> None
        天线的位置在时间轴上演进一个单位
    """

    def __init__(self, position: list, angles: list, velocity: list, slant, **kwargs):

        if kwargs["Ant_type"] == "URA":
            self.ant_type = "URA"
            self.shape = kwargs["Shape"]
            self.num = self.shape[0] * self.shape[1]
        if kwargs["Ant_type"] == "ULA":
            self.ant_type = "ULA"
            self.delta = kwargs["Delta"]
            self.num = kwargs["Num"]

        self.position = np.array(position)
        self.slant = slant * np.pi / 180
        self.azimuth = angles[0] * np.pi / 180
        self.elevation = angles[1] * np.pi / 180
        self.velocity = np.array(velocity).reshape(
            [
                3,
            ]
        )

    def evolve(self):
        """
        天线在时间轴上演进一个单位
        :return: 更新后的天线位置
        """
        self.position += self.velocity * dt


class Satellite:
    """
    储存卫星属性的类

    成员：
    height:float
        卫星的轨道高度，单位m
    azimuth:float
        卫星位置在ecef坐标系下的方位角，单位rad
    elevation:float
        卫星位置在ecef坐标系下的仰角，单位rad
    vsat:float
        卫星的速度标量，单位m/s
    velocity:list[float]
        卫星的速度矢量，单位m/s
    ecef_coordinate:list[float]
        卫星在ecef坐标系下的坐标，单位m
    """

    def __init__(self, coord_type, height, azimuth, elevation):
        """
        参数：
        coord_type:str
            输入的卫星坐标类型，可选lla,ecef,轨道根数，目前仅支持lla
        height:float
            lla坐标系下卫星位置的轨道高度，单位m
        azimuth:float
            lla坐标系下卫星位置的方位角，单位deg
        elevation:float
            lla坐标系下卫星位置的仰角，单位deg
        """
        if coord_type == "lla":
            # TODO:增加ecef和轨道根数的表示
            # TODO:增加卫星轨迹的演进
            self.height = height
            self.azimuth = azimuth * np.pi / 180
            self.elevation = elevation * np.pi / 180
            earth_mass = 5.972e24
            self.vsat = np.sqrt(
                scipy.constants.gravitational_constant * earth_mass / (self.height + Re)
            )
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
