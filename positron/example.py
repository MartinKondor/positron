# Learning the XOR gate with a neural network
import time

import numpy as np
import matplotlib.pyplot as plt

import positron.prep as prep
import positron.loss as loss
import positron.deep as deep


def get_data():
    return np.array([[0, 0],[0, 1],[1, 0],[1, 1]]), np.array([[0],[1],[1],[0]])


def run_classification():
    from sklearn.datasets import load_breast_cancer
    from sklearn.impute import SimpleImputer
    

    # Load the dataset
    deep.seed(1)
    data = load_breast_cancer()
    X = np.array(data.data)
    y = np.array([data.target]).T

    # Preprocess data
    y = prep.one_hot_encode(y)

    # Impute if there are empty fields
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(X)

    # Split data into separate arrays
    xtrain, xtest, ytrain, ytest = prep.split_train_test(X, y, shuffle=True)
    print("Data arrays:", end=" ")
    for _ in (xtrain, xtest, ytrain, ytest,):
        print(_.shape, end=" ")
    print()

    # Create the network
    layer_sizes = [30, 30, 2]
    layer_activ = ["relu", "relu", "sigmoid"]
    ws, bs = deep.init_network(X.shape[1], layer_sizes)
    print("layer_sizes:", layer_sizes)
    print("layer_activ:", layer_activ)

    # Start training
    orig_ws0 = np.copy(ws[0][0])
    cost = "cross_entropy"

    ws, bs, cost_history = deep.SGD(
        xtrain, ytrain,
        ws, bs,
        activations=layer_activ,
        costf=cost,
        eta=0.01,
        epochs=100,
        verbose=False,
        little_verbose=True,
        cost_history_needed=True
    )

    print(orig_ws0 - ws[0][0])
    o = deep.feedforward(xtest, ws, bs, layer_activ)
    print("Outputs:", *o[:5])
    print("Output should be:", *y[:5])
    print("Accuracy", 100 * round((np.sum(loss.mse(o, ytest)) / np.sum(y)), 2), "% for", len(ytest), "samples")
    costf = deep.get_cost_from_string(cost, return_derivate=False)

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


def run_xor():

    # Training data
    deep.seed(12)
    X, y = get_data()

    # Activation functions
    layer_sizes = [4, 4, 1]
    actifs = ["relu", "relu", "sigmoid"]

    # Generate a basic network
    weights, biases = deep.init_network(n_features=X.shape[1], weight_sizes=layer_sizes)
    
    # Hyperparameters 
    epochs = 4_500
    cost = "mse"

    # (v1.1) Time of training: 1.2309s for 4 rows and 10000 epochs
    orig_ws0 = np.copy(weights[0][0])

    start_time = time.time()
    weights, biases, cost_history = deep.SGD(
        X, y,
        weights, biases,
        activations=actifs,
        costf=cost,
        epochs=epochs,
        eta=0.9,
        mini_batch_size=1,
        verbose=False,
        little_verbose=True,
        cost_history_needed=True)
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
    print(orig_ws0 - weights[0][0])
    print("Example input/output:")
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    print("\tX:\t", *X)
    o = deep.feedforward(X, weights, biases, actifs)
    print("\ty_hat:\t", [int(_[0]) for _ in np.round(o)])
    print("\ty:\t", [int(_[0]) for _ in y])
    print("\tNeural Network outputs:\n\t", *o)


if __name__ == "__main__":
    #run_xor()
    run_classification()
