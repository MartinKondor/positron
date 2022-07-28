# Learning the XOR gate with a neural network
import numpy as np
from regex import P
import activ
import deep as d
import score

import matplotlib.pyplot as plt


def run():

    # Get some data
    d.seed(12)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([[0, 1, 1, 0]]).T

    # Activation functions & their derivatives
    layer_sizes = [4, 8, 1]
    actifs = [activ.sigmoid for _ in range(len(layer_sizes)+1)]
    dactifs = [activ.dsigmoid for _ in range(len(layer_sizes)+1)]

    # Generate a basic network
    d.seed(0)
    ws, bs = d.init_network(input_shape=X.shape, weight_sizes=layer_sizes, verbose=True)
    o = d.feedforward(X, ws, bs, actifs)
    print()
    print("Output of the network before the training:")
    print(o)
    print()
    
    # Hyperparameters 
    eta = 0.1
    epoch = 60_000
    cost = score.mse
    dcost = score.dmse
    mini_batch_size = 1
    ws, bs, cost_history = d.SGD(X, y, ws, bs, actifs, dactifs, cost, dcost, epoch, eta, mini_batch_size, verbose=False)

    o = d.feedforward(X, ws, bs, actifs)
    print()
    print("Output of the network after the training:")
    print(o)
    print()


if __name__ == "__main__":
    run()
