import numpy as np

import scoring
import calculating as calc


"""
Linear regression
"""
def linear(xtrain, ytrain, xtest, epochs=1000, learning_rate=0.000001):
    
    print(xtrain.shape, ytrain.shape)

    # Least Squares
    a = ytrain.mean()
    b = xtrain.mean()

    return a + xtest*b
