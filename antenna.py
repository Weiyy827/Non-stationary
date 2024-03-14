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

        self.position = position
        self.slant = slant
        self.azimuth = angles[0]
        self.elevation = angles[1]
        self.velocity = velocity


class Satellite:
    def __init__(self, height, azimuth, elevation):
        self.height = height
        self.azimuth = azimuth
        self.elevation = elevation
        self.vsat = np.sqrt(9.8 / self.height)
        self.Global_GCS_coordinate = np.array([(self.height + Re) * np.cos(self.elevation) * np.cos(self.azimuth),
                                               (self.height + Re) * np.cos(self.elevation) * np.sin(self.azimuth),
                                               (self.height + Re) * np.sin(self.elevation)])


class Origin:
    def __init__(self, latitude, longitude):
        self.longitude = longitude
        self.latitude = latitude
        self.Global_GCS_coordinate = np.array([Re * np.cos(self.latitude) * np.cos(self.longitude),
                                               Re * np.cos(self.latitude) * np.cos(self.longitude),
                                               Re * np.sin(self.latitude)])
