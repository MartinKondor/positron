# Learning the XOR gate with a neural network
import numpy as np
from regex import P
import activ
import deep as d
import score

import matplotlib.pyplot as plt


def run():

    # Get some data
    d.seed(63)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0, 1, 1, 0]]).T

    # Activation functions & their derivatives
    actifs = [activ.sigmoid, activ.sigmoid, activ.sigmoid]
    dactifs = [activ.dsigmoid, activ.sigmoid, activ.sigmoid]

    # Generate a basic network
    d.seed(0)
    ws, bs = d.init_network(input_shape=X.shape, weight_sizes=[8, 8], output_size=1, verbose=True)
    o = d.feedforward(X, ws, bs, actifs)
    print()
    print("Output of the network before the training:")
    print(o)
    print()

    print("Training", end="...")
    
    # Hyperparameters 
    eta = 0.01
    epoch = 50000
    cost = score.mse
    dcost = score.dmse
    ws, bs, cost_history = d.SGD(X, y, ws, bs, actifs, dactifs, cost, dcost, epoch, eta, verbose=False)
    print("done")

    o = d.feedforward(X, ws, bs, actifs)
    print("Output of the network after the training:")
    print(o)
    print()

    plt.grid()
    plt.plot(range(len(cost_history)), cost_history, c="b", linewidth=3)
    plt.show()


if __name__ == "__main__":
    run()
