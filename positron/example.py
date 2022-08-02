# Learning the XOR gate with a neural network
import time

import numpy as np
import matplotlib.pyplot as plt

import activ
import deep
import loss


def get_data():
    return np.array([[0, 0],[0, 1],[1, 0],[1, 1]]), np.array([[0],[1],[1],[0]])


def run():

    # Training data
    deep.seed(12)
    X, y = get_data()

    # Activation functions
    layer_sizes = [4, 4, 1]
    actifs = ["sigmoid", "sigmoid", "sigmoid"]

    # Generate a basic network
    weights, biases = deep.init_network(n_features=X.shape[1], weight_sizes=layer_sizes)
    
    # Hyperparameters 
    epochs = 4_500
    cost = "cross_entropy"

    # (v1.1) Time of training: 1.2309s for 4 rows and 10000 epochs

    start_time = time.time()
    weights, biases, cost_history = deep.SGD(
        X, y,
        weights, biases,
        activations=actifs,
        costf=cost,
        epochs=epochs,
        eta=0.9,
        mini_batch_size=1,
        verbose=True,
        cost_history_needed=False)
    end_time = time.time()

    o = deep.feedforward(X, weights, biases, actifs)
    costf = deep.get_cost_from_string(cost, return_derivate=False)
    print("Time of training:", str(round(end_time - start_time, 4)) + "s", "for", len(X), "rows and", epochs, "epochs")

    # Plot cost history
    if len(cost_history) > 0:
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
    print("\tX:\t", *X)
    o = deep.feedforward(X, weights, biases, actifs)
    print("\ty_hat:\t", [int(_[0]) for _ in np.round(o)])
    print("\ty:\t", [int(_[0]) for _ in y])
    print("\tNeural Network outputs:\n\t", *o)


if __name__ == "__main__":
    run()
