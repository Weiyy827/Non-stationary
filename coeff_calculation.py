import numpy as np


def field(elevation, azimuth, slant):
    elevation_deg = elevation * np.pi / 180
    azimuth_deg = azimuth * np.pi / 180
    vertical_cut = -np.min([12 * ((elevation_deg - 90) / 65) ** 2, 30])
    horizontal_cut = -np.min([12 * (azimuth_deg / 65) ** 2, 30])
    radiation_field = -np.min([-vertical_cut - horizontal_cut, 30])
    power_pattern = np.array(
        [
            np.sqrt(db2pow(radiation_field)) * np.cos(slant),
            np.sqrt(db2pow(radiation_field)) * np.sin(slant),
        ]
    ).reshape([2, 1])
    return power_pattern


def db2pow(db):
    return 10 ** (db / 10)


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
