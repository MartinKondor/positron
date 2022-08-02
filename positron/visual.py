import numpy as np
from matplotlib import pyplot as plt


def _vis_basic(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot"):
    X = np.linspace(lowerx, upperx, num=len(y))
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    return X, fig


def plot(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", grid=True, **kwargs):
    X, fig = _vis_basic(y, lowerx, upperx, figsize, title)
    if grid:
        plt.grid()
    plt.plot(X, y, **kwargs, linewidth=2)
    return X, fig


def show_plot(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", grid=True, **kwargs):
    X, fig = plot(y, lowerx, upperx, figsize, title, grid, **kwargs)
    plt.show()


def scatter(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", grid=True, **kwargs):
    X, fig = _vis_basic(y, lowerx, upperx, figsize, title)
    if grid:
        plt.grid()
    plt.scatter(X, y, **kwargs, linewidth=2)
    return X, fig


def show_scatter(y, lowerx=-10, upperx=10, figsize=(7, 7,), title="Plot", grid=True, **kwargs):
    X, fig = scatter(y, lowerx, upperx, figsize, title, grid, **kwargs)
    plt.show()


if __name__ == "__main__":
    import activ
    x = np.arange(-10, 10, 0.1)
    show_plot(activ.softplus(x), grid=True)
