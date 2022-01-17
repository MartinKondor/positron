import numpy as np
import math


"""
:A: square matrix (np.ndarray)
:returns: determinant of A
"""
def det(A: np.ndarray):
    if A.shape[0] < 2 or A.shape[0] != A.shape[1]:
        raise Exception('Wrong matrix dimensions for determinant')

    n = A.shape[0]

    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    a = np.copy(A)
    s = 0

    for i in range(n):
        s += (-1 if i%2!=0 else 1) * a[0][i] * det(np.delete(np.delete(a, 0, 0), i, 1))    
        a = np.copy(A)

    return s


"""
:returns: matrix A with chess table rule applied
"""
def chess_table_rule(A: np.ndarray):
    if A.shape[0] < 2 or A.shape[0] != A.shape[1]:
        raise Exception('Wrong matrix dimensions for chess table rule')

    a = np.copy(A)
    n = len(A)
    c = 0

    for i in range(n):
        for j in range(n):
            if c%2!=0:
                a[i][j] *= -1
            c += 1

    return a


"""
:returns: subdeterminant of matrix A (minor matrix)
"""
def subdet(A: np.ndarray):
    if A.shape[0] < 3 or A.shape[0] != A.shape[1]:
        raise Exception('Wrong matrix dimensions for adjungate')

    n = len(A.shape)
    a = np.copy(A)
    na = []

    for i in range(n + 1):
        row = []
        for j in range(n + 1):
           row.append(det(np.delete(np.delete(a, i, 0), j, 1))) 
        na.append(row)

    return np.array(na)


"""
:A: square matrix (np.ndarray)
:returns: adjungate of A
"""
def adj(A: np.ndarray):
    return chess_table_rule(subdet(A)).T


def inv(A: np.ndarray):
    d = det(A)
    if d == 0:
        raise Exception('Matrix cannot be inverted')
    return adj(A) / d


"""
Convert data from cartesian coorinates to polar coordinates.
:returns: New X and Y values
"""
def to_polar(X: np.ndarray, Y: np.ndarray):
    nX, nY = [], []

    for x, y in zip(X, Y):
        nX.append((x**2 + y**2)**0.5)
        if x != 0:
            nY.append(math.degrees(math.atan(y / x)))
        else:
            nY.append(0)

    """
    # For visual testing
    import matplotlib.pyplot as plt
    
    plt.scatter(X, Y, c="red")
    plt.scatter(nX, nY, c="green")
    plt.show()
    """
    return np.array(nX), np.array(nY)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


if __name__ == '__main__':
    import random


    print('Testing', __file__)

    A = np.array([
        [1, 1, 3],
        [2, 3, -4],
        [3, -2, 5],
    ])

    print(det(A))

    B = np.array([
        [1, -2, 0],
        [2, -1, 1],
        [-1, 0, -3]
    ])

    C = np.array([
        [1, 2],
        [3, 4]
    ])
    
    # Test data
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([random.randint(10, 100) for _ in range(10)])

    print(chess_table_rule(C))
    print(adj(B))
    print(inv(B))

    print(X, y)
    print(to_polar(X, y))
