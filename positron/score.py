import numpy as np


"""
Residual Sum of Squares
"""
def rss(y_hat, y):
    return (y - y_hat)**2

def drss(y_hat, y):
    return -2*y + 2*y_hat


"""
Mean Square Error
"""
def mse(y_hat, y):
    return  1/2 * (y - y_hat)**2

def dmse(y_hat, y):
    return y_hat - y


if __name__ == "__main__":
    y_hat1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_hat2 = np.array([0, 2, 4, 7, 15, 32, 59, 128])
    y_hat3 = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    y = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    
    print("y_hat1 =\t", *y_hat1)
    print("y_hat2 =\t", *y_hat2)
    print("y_hat3 =\t", *y_hat3)
    print("y =\t\t", *y)
    print()
    print("mse y_hat1:\t", *mse(y_hat1, y))
    print("mse y_hat2:\t", *mse(y_hat2, y))
    print("mse y_hat3:\t", *mse(y_hat3, y))
    print()
    print("rss y_hat1:\t", *rss(y_hat1, y))
    print("rss y_hat2:\t", *rss(y_hat2, y))
    print("rss y_hat3:\t", *rss(y_hat3, y))
