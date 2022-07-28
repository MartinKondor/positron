import numpy as np


"""
Sigmoid function
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.arange(-10, 10, 0.1)
    plt.title("Activation functions")
    plt.grid()
    plt.plot(X, sigmoid(X), c="g", linewidth=3)
    plt.plot(X, dsigmoid(X), c="r", linewidth=3)
    plt.legend(["Sigmoid", "Sigmoid derivate"])
    plt.show()

