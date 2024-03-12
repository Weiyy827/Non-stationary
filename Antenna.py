class Antenna:
    def __init__(self, position: list, angles: list, velocity: list, **kwargs):

        if kwargs['Ant_type'] == 'URA':
            self.Ant_type = 'URA'
            self.shape = kwargs['Shape']
            self.num = self.shape[0] * self.shape[1]
        if kwargs['Ant_type'] == 'ULA':
            self.Ant_type = 'ULA'
            self.delta_Ant = kwargs['Delta']
            self.num = kwargs['Num']

        self.position = position
        self.azimuth = angles[0]
        self.elevation = angles[1]
        self.velocity = velocity
