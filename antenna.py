import numpy as np

from config import Re


class Antenna:
    def __init__(self, position: list, angles: list, velocity: list, slant, **kwargs):

        if kwargs['Ant_type'] == 'URA':
            self.Ant_type = 'URA'
            self.shape = kwargs['Shape']
            self.num = self.shape[0] * self.shape[1]
        if kwargs['Ant_type'] == 'ULA':
            self.Ant_type = 'ULA'
            self.delta_Ant = kwargs['Delta']
            self.num = kwargs['Num']

        self.position = np.array(position)
        self.slant = slant * np.pi / 180
        self.azimuth = angles[0] * np.pi / 180
        self.elevation = angles[1] * np.pi / 180
        self.velocity = velocity


class Satellite:
    def __init__(self, height, azimuth, elevation):
        self.height = height
        self.azimuth = azimuth * np.pi / 180
        self.elevation = elevation * np.pi / 180
        self.vsat = np.sqrt(9.8 / self.height)
        self.velocity = np.array([self.vsat * np.cos(self.azimuth), self.vsat * np.sin(self.azimuth), 0]).reshape([1,3])
        self.Global_GCS_coordinate = np.array([(self.height + Re) * np.cos(self.elevation) * np.cos(self.azimuth),
                                               (self.height + Re) * np.cos(self.elevation) * np.sin(self.azimuth),
                                               (self.height + Re) * np.sin(self.elevation)])


class Origin:
    def __init__(self, latitude, longitude):
        self.longitude_deg = longitude
        self.latitude_deg = latitude

        self.longitude_rad = longitude * np.pi / 180
        self.latitude_rad = latitude * np.pi / 180
        self.Global_GCS_coordinate = np.array([Re * np.cos(self.latitude_rad) * np.cos(self.longitude_rad),
                                               Re * np.cos(self.latitude_rad) * np.sin(self.longitude_rad),
                                               Re * np.sin(self.latitude_rad)])

        z = (self.longitude_deg - 90) * np.pi / 180  # z：z轴不动逆时针旋转经度-90
        y = (90 - self.latitude_deg) * np.pi / 180  # y: y轴不动顺时针时针旋转90-纬度
        self.Rotation = np.array([np.cos(z), np.sin(z), 0,
                                  -np.sin(z), np.cos(z), 0,
                                  0, 0, 1]).reshape([3, 3]) @ np.array([np.cos(y), 0, -np.sin(y),
                                                                        0, 1, 0,
                                                                        np.sin(y), 0, np.cos(y)]).reshape([3, 3])
