import numpy as np

a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
b = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

c = np.convolve(a, b)
print(c)
