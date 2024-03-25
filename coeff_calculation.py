import numpy as np

from utils import db2pow


def field(zenith, azimuth, slant):
    vertical_cut = -np.min([12 * ((zenith - 90) / 65) ** 2, 30])
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
