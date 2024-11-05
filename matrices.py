import numpy as np
import vectors as v
import random
from typing import List


def get_matrix(n: int, m: int) -> np.ndarray:
    matrix = np.array([[random.random() for _ in range(m)] for _ in range(n)], dtype=np.float64)
    return matrix

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError("Matrices must have the same dimensions for addition.")
    return x + y

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    return x * a


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape[1] != y.shape[0]:
        raise ValueError("The number of columns in the first matrix must equal the number of rows in the second matrix.")
    return np.array([[sum(x[i][k] * y[k][j] for k in range(len(y))) for j in range(len(y[0]))] for i in range(len(x))])


def identity_matrix(dim: int) -> np.ndarray:
    # Initialize a dim x dim matrix filled with zeros
    identity = np.zeros((dim, dim), dtype=np.float64)

    # Set the diagonal elements to 1
    for i in range(dim):
        identity[i, i] = 1.0

    return identity


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    n = len(x)
    augmented_matrix = np.hstack((x, np.eye(n, dtype=np.float64)))
    for i in range(n):
        if augmented_matrix[i, i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        augmented_matrix[i] /= augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]
    return augmented_matrix[:, n:]

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    return x.T


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError("Matrices must have the same dimensions for the Hadamard product.")

    # Initialize an empty matrix with the same shape as x and y
    result = np.zeros_like(x, dtype=np.float64)

    # Element-wise multiplication
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j] * y[i, j]

    return result

def basis(x: np.ndarray) -> list:
    n_rows, n_cols = x.shape
    x = x.astype(np.float64)
    pivot_columns = []
    row = 0
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if x[r, col] != 0:
                pivot_row = r
                break
        if pivot_row is not None:
            if pivot_row != row:
                x[[row, pivot_row]] = x[[pivot_row, row]]
            x[row] /= x[row, col]
            for r in range(n_rows):
                if r != row:
                    x[r] -= x[row] * x[r, col]
            pivot_columns.append(col)
            row += 1
    return pivot_columns

def norm(x: np.ndarray, order: int | float | str = 'fro') -> float:
    if order == 'fro':
        return np.sqrt(np.sum(x**2))
    elif order == 2:
        _, s, _ = np.linalg.svd(x)
        return s[0]
    elif order == float('inf'):
        return np.max(np.sum(np.abs(x), axis=1))
    else:
        raise ValueError(f"Unsupported norm order: {order}")
