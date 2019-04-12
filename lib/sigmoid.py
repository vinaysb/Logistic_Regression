import numpy as np
from math import exp


def sigmoid(z):
    if (np.isscalar(z)):
        g = 1 / (1 + np.exp(-z))
    elif (z.ndim == 1):
        g = np.zeros((np.size(z), 1))
        for i in range(0, np.size(z, 0)):
            g[i] = 1 / (1 + np.exp(-z[i]))
    else:
        g = np.zeros(z.shape)
        for i in range(0, np.size(z, 0)):
            for j in range(0, np.size(z, 1)):
                g[i, j] = 1 / (1 + exp(-z[i, j]))
    return g
