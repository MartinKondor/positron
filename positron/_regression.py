import numpy as np

import calculating as calc


"""
:X: np.ndarray X axsis data
:y: np.ndarray y axsis data
:returns: values for the regression line
"""
def linear(trainX, trainy, testX=None):
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


"""
Logistic regression for binary classification (0, or 1)
:returns: y with classes
"""
def binary_logistic(trainXs, trainy, is_inclusive=True):
    r = calc.sigmoid(trainXs[0] * trainy.T)
    
    for i in range(1, len(trainXs)):
        r += calc.sigmoid(trainXs[i] * trainy.T)
    
    r /= len(trainXs)
    r[r > 0.5] = 1
    r[r < 0.5] = 0
    if is_inclusive:
        r[r == 0.5] = 0
    else:
        r[r == 0.5] = 1
    return r


"""
Logistic regression for binary classification (0, or 1)
:returns: y with classes
"""
def multinomial_logistic(trainXs, trainy):
    classes = np.unique(trainy)
    has_0_class = False
    
    if 0 in list(classes):
        has_0_class = True
        classes = np.array(classes) + 1

    border = 1 / len(classes)
    r = calc.sigmoid(trainXs[0] * trainy.T)
    
    for i in range(1, len(trainXs)):
        r += calc.sigmoid(trainXs[i] * trainy.T)
    
    r /= len(trainXs)
    
    for i in range(len(r)):
        print(r[i], trainy[i])

    for i, cls in enumerate(classes):
        r[(i*border < r) & (r < (i + 1)*border)] = cls
        if i == 0:
            r[r == 0] = cls
        r[r == (i + 1)*border] = cls

    if has_0_class:
        r = r - 1
    return r
