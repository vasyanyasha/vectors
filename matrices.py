import numpy as np
import vectors as v
import random
from typing import List


def get_matrix(n: int, m: int) -> List[List[float]]:
    matrix = []

    for i in range(n):
        row = [random.random() for _ in range(m)]
        matrix.append(row)

    return matrix


def add(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    if len(x) != len(y) or len(x[0]) != len(y[0]):
        raise ValueError("Matrices must have the same dimensions for addition.")

    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[0])):
            row.append(x[i][j] + y[i][j])
        result.append(row)

    return result


def scalar_multiplication(x: List[List[float]], a: float) -> List[List[float]]:
    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[0])):
            row.append(x[i][j] * a)
        result.append(row)

    return result


def dot_product(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    if len(x[0]) != len(y):
        raise ValueError(
            "The number of columns in the first matrix must equal the number of rows in the second matrix.")

    result = [[0 for _ in range(len(y[0]))] for _ in range(len(x))]

    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]

    return result


def identity_matrix(dim: int) -> List[List[float]]:
    identity = [[0 for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        identity[i][i] = 1.0

    return identity


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    n = len(x)

    augmented_matrix = [list(map(float, x[i])) + [1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):
        if augmented_matrix[i][i] == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        pivot = augmented_matrix[i][i]
        for j in range(2 * n):
            augmented_matrix[i][j] = augmented_matrix[i][j] / float(pivot)

        for k in range(n):
            if i != k:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]

    return inverse_matrix


def matrix_transpose(x: List[List[float]]) -> List[List[float]]:
    rows = len(x)
    cols = len(x[0])

    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = x[i][j]

    return transpose


def hadamard_product(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    if len(x) != len(y) or len(x[0]) != len(y[0]):
        raise ValueError("Matrices must have the same dimensions for the Hadamard product.")

    # Element-wise multiplication
    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[0])):
            row.append(x[i][j] * y[i][j])
        result.append(row)

    return result


def basis(x: List[List[float]]) -> List[int]:
    n_rows = len(x)
    n_cols = len(x[0])

    matrix = [list(map(float, row)) for row in x]

    pivot_columns = []

    row = 0
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if matrix[r][col] != 0:
                pivot_row = r
                break

        if pivot_row is not None:
            if pivot_row != row:
                matrix[row], matrix[pivot_row] = matrix[pivot_row], matrix[row]

            # Normalize the pivot row
            pivot_val = matrix[row][col]
            for j in range(n_cols):
                matrix[row][j] /= pivot_val

            # Eliminate the current column from all other rows
            for r in range(n_rows):
                if r != row:
                    factor = matrix[r][col]
                    for j in range(n_cols):
                        matrix[r][j] -= factor * matrix[row][j]

            # Track the pivot column
            pivot_columns.append(col)

            # Move to the next row
            row += 1

    return pivot_columns


def norm(x: List[List[float]], order: int | float | str = 'fro') -> float:
    if order == 'fro':
        # Frobenius norm: sqrt of sum of all elements squared
        return sum(sum(x[i][j] ** 2 for j in range(len(x[0]))) for i in range(len(x))) ** 0.5

    elif order == 2:


        matrix_np = np.array(x)

        u, s, vh = np.linalg.svd(matrix_np)

        return s[0]  # Largest singular value

    elif order == float('inf'):
        # Max norm: Maximum absolute row sum
        return max(sum(abs(x[i][j]) for j in range(len(x[0]))) for i in range(len(x)))

    else:
        raise ValueError(f"Unsupported norm order: {order}")

