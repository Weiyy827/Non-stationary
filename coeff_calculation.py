import numpy as np


def field(elevation, azimuth, slant):
    elevation_deg = elevation * np.pi / 180
    azimuth_deg = azimuth * np.pi / 180
    Vertical_cut = -np.min([12 * ((elevation_deg - 90) / 65) ** 2, 30])
    Horizontal_cut = -np.min([12 * (azimuth_deg / 65) ** 2, 30])
    Radiation_field = -np.min([-Vertical_cut - Horizontal_cut, 30])
    Power_pattern = np.array([np.sqrt(db2pow(Radiation_field)) * np.cos(slant),
                              np.sqrt(db2pow(Radiation_field)) * np.sin(slant)]).reshape([2, 1])
    return Power_pattern


def db2pow(db):
    return 10 ** (db / 10)


def cross_polar(Xnm, Phase):
    XPR = 10 ** (Xnm / 10)
    Cross_polar = np.array([np.exp(1j * Phase[0]), np.sqrt(XPR ** -1) * np.exp(1j * Phase[1]),
                            np.sqrt(XPR ** -1) * np.exp(1j * Phase[2]), np.exp(1j * Phase[3])]).reshape([2,2])
    return Cross_polar
