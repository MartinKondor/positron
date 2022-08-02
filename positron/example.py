# Learning the XOR gate with a neural network
import time
import math

import matplotlib.pyplot as plt

import numpy as np
from regex import P
import activ
import deep as d
import score


def run():

    # Get training data
    d.seed(12)
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    y = np.array([[0],[1],[1],[0]])

    # Activation functions
    layer_sizes = [4, 4, 1]
    actifs = ["sigmoid", "sigmoid", "sigmoid"]

    # Generate a basic network
    ws, bs = d.init_network(input_shape=X.shape, weight_sizes=layer_sizes, verbose=False)
    
    # Hyperparameters 
    epochs = 10_000
    cost = "mse"

    start_time = time.time()
    ws, bs, cost_history = d.SGD(
        X, y, ws, bs,
        activations=actifs,
        costf="mse",
        epochs=epochs,
        eta=0.999,
        mini_batch_size=1,
        verbose=True,
        cost_history_needed=False)
    end_time = time.time()

    o = d.feedforward(X, ws, bs, actifs)
    costf = d.get_cost_from_string(cost, return_derivate=False)
    print("Time of training:", str(round(end_time - start_time, 4)) + "s", "for", len(X), "rows and", epochs, "epochs")

    # Plot cost history
    if False and len(cost_history) > 0:
        if len(cost_history) > 100:
            cost_history = cost_history[:100]

        print("Smallest cost =", round(np.min(cost_history), 4))

        plt.title("Cost history over time")
        plt.xlabel("Time")
        plt.ylabel("Cost (loss)")
        plt.legend(["Cost"])
        plt.grid()
        plt.plot(range(len(cost_history)), cost_history, c="g", alpha=0.9)
        plt.show()
    else:
        print("Cost =", round(costf(o, y).sum(), 4))

    print()
    print("Example input/output:")
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    print("\t", "Inputs:\t\t", *X)
    o = d.feedforward(X, ws, bs, actifs)
    print("\t", "Outputs:\t", [int(_[0]) for _ in np.round(o)])
    print("\t", "Desired outputs:", [int(_[0]) for _ in y])
    print("\t", "Neural Network outputs:")
    print("\t", *o)


if __name__ == "__main__":
    run()
