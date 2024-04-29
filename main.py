from scipy import constants
from scipy.io import savemat

from src.channel_model import non_stationary_channel
from src.simpar import Simulation_parameter, Satellite, Antenna
from src.utils import ecef2gcs

# 参数设置
simpar = Simulation_parameter()

simpar.fc = 2e9  # 载波频率，单位GHz
simpar.sat = Satellite('lla', height=500e3, azimuth=0, elevation=90)
original = [0, 90]

simpar.rx = Antenna([0, 0, 1.5],
                    [0, 90],
                    [0.5, 0, 0],
                    45,
                    Ant_type="ULA",
                    Num=32,
                    Delta=constants.c / simpar.fc / 2)

simpar.tx = Antenna(ecef2gcs(simpar.sat.ecef_coordinate, original),
                    [0, -90],
                    ecef2gcs(simpar.sat.velocity, original),
                    45,
                    Ant_type="ULA",
                    Num=32,
                    Delta=constants.c / simpar.fc / 2, )

# 过信道
simpar.snapshots = 1
LOS_coeff, NLOS_coeff, Cluster_set = non_stationary_channel(simpar)
savemat('channel_coeff.mat', {'LOS_coeff': LOS_coeff, 'NLOS_coeff': NLOS_coeff, 'Cluster_set': Cluster_set})
