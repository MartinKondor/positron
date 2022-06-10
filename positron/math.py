import numpy as np
import math as m


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
    if A.shape[0] != A.shape[1]:
        raise Exception('Wrong matrix dimensions for adjungate')
    return chess_table_rule(subdet(A)).T


"""
Convert data from cartesian coorinates to polar coordinates.
:returns: New X and Y values
"""
def to_polar(X: np.ndarray, Y: np.ndarray):
    nX, nY = [], []

    for x, y in zip(X, Y):
        nX.append((x**2 + y**2)**0.5)
        if x != 0:
            nY.append(m.degrees(m.atan(y / x)))
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


"""
Determines if a matrix is singular (has no inverse) or not.
:returns: True or False
"""
def is_singular(A: np.ndarray):
    return det(A) == 0


"""
:returns: the inverse of the given matrix
"""
def inv(A: np.ndarray):
    if is_singular(A):
        raise Exception('Matrix cannot be inverted')
    return adj(A) / det(A)


"""
:returns: the Moore Penrose Inverse of a matrix
"""
def moore_penrose_inv(X):
    return chess_table_rule(X.T)


"""
:reutrns: the LP norm of the given data
"""
def norm(A: np.ndarray, p: int):
    return np.sum(A**p)**(1/p)


"""
:returns: n x n identity matrix
"""
def I(n: int):
    A = []
    for i in range(n):
        row = [0 for j in range(n)]
        row[i] = 1
        A.append(row)
    return np.array(A) 


"""
Trace operator
:returns: the sum of the diagonal entries
"""
def tr(A: np.ndarray):
    vec = []
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            if i == j:
                vec.append(A[i][j])
    return np.sum(np.array(vec))


if __name__ == '__main__':
    pass
