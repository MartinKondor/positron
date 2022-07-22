# Learning the XOR gate with a neural network
import numpy as np
import activ
import deep as d


def run():

    # Get some data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0, 1, 1, 0]]).T

    # Hyperparameters 
    eta = 0.01
    epoch = 1

    # Activation functions & their derivatives
    actifs = [activ.sigmoid, activ.sigmoid, activ.sigmoid]
    dactifs = [activ.dsigmoid, activ.sigmoid, activ.sigmoid]

    # Generate a basic network
    d.seed(0)
    ws, bs = d.init_network(input_shape=X.shape, weight_sizes=[4, 4], output_size=1, verbose=True)
    o = d.feedforward(X, ws, bs, actifs)
    print()
    print("Output of the network before the training:")
    print(o)
    print()

    print("Training:")
    ws, bs, cost_history = d.train(X, y, ws, bs, actifs, dactifs, epoch, eta)


if __name__ == "__main__":
    run()
