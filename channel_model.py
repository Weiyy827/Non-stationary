import numpy as np

import calculation
from cluster_evolution import cluster_evolution


def non_stationary_channel(inputWaveform, numTx, numRx, fc):
    cluster_evolution()
    # 计算时延
    delay = []
    # 信道系数
    coeff = calculation.calculation_coeff()
    # 系统函数，在时延位置为信道系数，其余位置为0
    h = calculation.calculation_h(coeff, delay)
    outputWaveform = np.zeros([numTx, len(inputWaveform)])
    for i in range(numTx):
        outputWaveform[i] = np.convolve(inputWaveform[i], h)

    return None
