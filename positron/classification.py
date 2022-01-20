

from sklearn import neighbors
import numpy as np


class KNearestNeighbor:

    def __init__(self, xtrain, ytrain, k=2):
        self.k = k

    def predict(self, xtest):
        neighbors = xtest
        return 1/self.k * neighbors.sum()
