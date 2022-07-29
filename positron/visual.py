import numpy as np
from matplotlib import pyplot as plt


def _vis_basic(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot"):
    X = np.linspace(lowerx, upperx, num=len(y))
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    return X, fig


def plot(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", c="b"):
    X, fig = _vis_basic(y, lowerx, upperx, figsize, title)
    plt.plot(X, y, c=c, linewidth=2)
    return X, fig


def show_plot(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", c="b"):
    X, fig = plot(y, lowerx, upperx, figsize, title, c)
    plt.show()


def scatter(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", c="b", marker="o"):
    X, fig = _vis_basic(y, lowerx, upperx, figsize, title)
    plt.plot(X, y, c=c, marker=marker, linewidth=2)
    return X, fig


def show_scatter(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", c="b", marker="o"):
    X, fig = scatter(y, lowerx, upperx, figsize, title, c, marker)
    plt.show()


if __name__ == "__main__":
    show_plot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
