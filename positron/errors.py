import numpy as np

"""
Root Mean Square Error
"""
def rmse(y_hat: np.ndarray, y: np.ndarray):
    return  (((y_hat - y)**2).sum() / y_hat.shape[0])**.5
 

"""
Mean Absolute Error
"""
def mae(y_hat: np.ndarray, y: np.ndarray):
    return np.abs((y_hat - y).sum()) / y_hat.shape[0]


if __name__ == "__main__":
    y_hat1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_hat2 = np.array([0, 2, 4, 7, 15, 32, 59, 128])
    y_hat3 = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    y = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    
    print("rmse y_hat1:", rmse(y_hat1, y))
    print("rmse y_hat2:", rmse(y_hat2, y))
    print("rmse y_hat3:", rmse(y_hat3, y))
    print()
    print("mae y_hat1:", mae(y_hat1, y))
    print("mae y_hat2:", mae(y_hat2, y))
    print("mae y_hat3:", mae(y_hat3, y))
