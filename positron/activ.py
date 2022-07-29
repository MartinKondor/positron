"""
WARNING: Every activation function should return a np.ndarray.
"""
from turtle import xcor
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def identity(z):
    return z

def didentity(z):
    return np.ones(z.shape)


def binary_step(z):
    r = np.copy(z)
    r[z < 0] = 0
    r[z >= 0] = 1
    return r

def dbinary_step(z):
    r = np.copy(z)
    r[z == 0] = 1
    r[z != 0] = 0
    return r


def tanh(z):
    return 2*sigmoid(2*z) - 1

def dtanh(z):
    return 1 - tanh(z)**2


def arctan(z):
    return np.arctan(z)

def darctan(z):
    return 1 / (1 + z**2)


def prelu(z, p):
    r = np.copy(z)
    r[z < 0] = p*r[z < 0]
    return r

def dprelu(z, p):
    r = np.copy(z)
    r[z < 0] = p
    r[z >= 0] = 1
    return r


def relu(z):
    return prelu(z, 0)

def drelu(z, p=1):
    return dprelu(z, 0)


def elu(z, p=1):
    r = np.copy(z)
    r[z < 0] = p*(np.exp(z) - 1)
    return r

def delu(z, p=1):
    r = np.copy(z)
    r[z < 0] = elu(z) + p
    r[z >= 0] = 1
    return r


def softplus(z, p=1):
    return np.log(1 + np.exp(z))

def dsoftplus(z, p=1):
    return sigmoid(z)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.arange(-8, 8, 0.1)
    fig = plt.figure(figsize=(12, 6,))
    axes = fig.subplots(nrows=2, ncols=3)
    function_names = ["sigmoid","binary_step","tanh","arctan","relu (+prelu, elu)","softplus"]
    functions = [sigmoid, binary_step, tanh, arctan, relu, softplus]
    dfunctions = [dsigmoid, dbinary_step, dtanh, darctan, drelu, dsoftplus]

    for i, (n, f, df,) in enumerate(zip(function_names, functions, dfunctions)):
        ax = axes[i//3][i%3]
        ax.set_title(n)
        ax.grid()

        ax.plot(X, f(X), c="b", linewidth=2)
        ax.plot(X, df(X), c="r", linewidth=2)

    plt.show()

