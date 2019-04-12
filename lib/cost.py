import numpy as np
from lib.sigmoid import sigmoid


def cost(main_theta, X, y, lmda):
    theta = np.copy(main_theta)
    theta = np.reshape(theta, (len(theta), 1))
    m = np.size(y, 0)
    J = 0

    h = sigmoid(X.dot(theta))
    theta[0, 0] = 0

    le = float((-(y.transpose())).dot(np.log(h)))
    ri = float((1 - (y.transpose())).dot(np.log(1 - h)))
    jreg = (lmda / (2 * m)) * float(theta.transpose().dot(theta))
    J = ((le - ri) / m) + jreg

    return J
