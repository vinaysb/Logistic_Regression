import numpy as np
from lib.sigmoid import sigmoid


def predict(main_theta, X):
    theta = np.copy(main_theta)
    m = np.size(X, 0)
    p = np.zeros((m, 1))
    for i in range(m):
        temp = sigmoid(theta.transpose().dot(X[i, :].transpose()))
        if temp >= 0.5:
            p[i] = 1
    return p
