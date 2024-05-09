"""信道模型主入口，计算当前条件下的信道系数

"""

from scipy import constants

from src.channel_model import non_stationary_channel
from src.simpar import SimulationParameter, Satellite, Antenna
from src.utils import ecef2gcs

if __name__ == "__main__":
    # 参数设置
    simpar = SimulationParameter()

    simpar.fc = 2e9  # 载波频率，单位Hz
    simpar.sat = Satellite(
        "lla", height=500e3, azimuth=0, elevation=90
    )  # 定义卫星对象，采用lla坐标系，单位角度
    simpar.original = [
        0,
        90,
    ]  # 定义GCS坐标系原点在地球上的位置，采用方位角和仰角定义，单位角度

    # 定义接收天线
    simpar.rx = Antenna(
        [0, 0, 1.5],
        [0, 90],
        [0.5, 0, 0],
        45,
        Ant_type="ULA",
        Num=32,
        Delta=constants.c / simpar.fc / 2,
    )

    # 定义发射天线
    simpar.tx = Antenna(
        ecef2gcs(simpar.sat.ecef_coordinate, simpar.original),
        [0, -90],
        ecef2gcs(simpar.sat.velocity, simpar.original),
        45,
        Ant_type="ULA",
        Num=32,
        Delta=constants.c / simpar.fc / 2,
    )

    # 定义仿真时刻数
    simpar.snapshots = 1

    # 定义LOS场景
    simpar.isLOS = False

    # 计算信道系数并保存
    non_stationary_channel(simpar)
