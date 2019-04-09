import numpy as np
from lib.cost import cost
from lib.predict import predict
from scipy.optimize import fmin_bfgs


def Logistic_Regression(features_train, features_test, labels_train):

    initial_theta = np.zeros((np.size(features_train, 1), 1))

    theta = fmin_bfgs(f=lambda theta: cost(theta, features_train, labels_train, 1), x0=initial_theta, maxiter=400, disp=False)

    pred = predict(theta, features_test)

    return pred
