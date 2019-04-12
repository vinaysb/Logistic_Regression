import numpy as np
from lib.sigmoid import sigmoid


def grad(main_theta, X, y, lmda):
    theta = np.copy(main_theta)
    m = np.size(y, 0)
    grad = np.zeros((np.size(theta), 1))

    h = sigmoid(X.dot(theta))
    theta[0, 0] = 0
    error = h - y
    gradreg = (lmda / m) * theta
    grad = (1 / m * (X.transpose().dot(error))) + gradreg
    return grad
