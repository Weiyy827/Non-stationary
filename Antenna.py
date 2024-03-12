class Antenna:
    def __init__(self, shape: list, position: list, angles: list, delta_Ant):
        self.shape = shape
        self.num = shape[0] * shape[1]
        self.position = position
        self.delta_Ant = delta_Ant
        self.azimuth = angles[0]
        self.elevation = angles[1]
