import numpy as np

import scoring
import calculating as calc


"""
Linear regression
"""
def linear(xtrain, ytrain, xtest, epochs=1000, learning_rate=0.000001):
    _slope = xtrain - xtrain.mean()
    slope = (_slope * (ytrain - ytrain.mean())).sum() / (_slope**2).sum()
    return (ytrain.mean() - slope*xtrain.mean()) + xtest * slope
