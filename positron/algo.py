import numpy as np


"""
:X: np.ndarray X axsis data
:y: np.ndarray y axsis data
:returns: values for the regression line
"""
def linear_regression(trainX, trainy, testX=None):
    xss = (trainX ** 2).sum()
    xs = trainX.sum()
    xys = (trainX * trainy).sum()
    ys = trainy.sum()

    bottom = (trainX.size * xss - xs**2)
    a = (ys * xss - xs * xys) / bottom
    b = (trainX.size * xys - xs * ys) / bottom
    
    if testX is None:
        testX = trainX
    return b*testX + a
