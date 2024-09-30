from typing import Sequence

import numpy as np
import random
import math
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    vector = [[random.random()] for _ in range(dim)]

    return np.array(vector, dtype=float)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    num_nonzeros = random.randint(1, (dim + 1) // 2)

    row_indices = random.sample(range(dim), num_nonzeros)

    values = [random.random() for _ in range(num_nonzeros)]

    return sparse.coo_matrix((values, (row_indices, [0] * num_nonzeros)), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError("Vectors must be of the same shape.")

    result = [[x[i][0] + y[i][0]] for i in range(len(x))]

    return np.array(result, dtype=float)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    return a * x


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    if len(vectors) != len(coeffs):
        raise ValueError("The number of vectors and coefficients must be the same.")

    result = np.zeros_like(vectors[0])

    for vec, coeff in zip(vectors, coeffs):
        result += coeff * vec

    return result

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Vectors must be of the same shape.")

    result = sum(x[i][0] * y[i][0] for i in range(len(x)))

    return result


def norm(x: np.ndarray, order: int | float) -> float:
    if order == 1:
        return sum(abs(x[i][0]) for i in range(len(x)))

    elif order == 2:
        return (sum(x[i][0] ** 2 for i in range(len(x)))) ** 0.5

    elif order == float('inf'):
        return max(abs(x[i][0]) for i in range(len(x)))

    else:
        raise ValueError("Unsupported norm order. Use 1, 2, or inf.")


def distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Vectors must be of the same shape.")

    squared_diff_sum = sum((x[i][0] - y[i][0])**2 for i in range(len(x)))

    return squared_diff_sum**0.5


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Vectors must be of the same shape.")

    # Calculate dot product
    dot_prod = dot_product(x, y)

    # Calculate norms
    norm_x = norm(x, 2)  # Euclidean norm
    norm_y = norm(y, 2)  # Euclidean norm

    # Calculate cosine of the angle
    cosine_angle = dot_prod / (norm_x * norm_y)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = angle_rad * (180 / np.pi)

    return angle_deg


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    # Compute the dot product
    if x.shape != y.shape:
        raise ValueError("Vectors must be of the same shape.")

        # Calculate dot product
    dot_prod = dot_product(x, y)

    # Check if the dot product is close to zero
    return math.isclose(dot_prod, 0)

def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)

    # Augment the matrix 'a' with vector 'b'
    aug_matrix = np.hstack((a.astype(np.float64), b.reshape(-1, 1).astype(np.float64)))

    # Forward elimination
    for i in range(n):
        # Ensure the pivot is non-zero
        if aug_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if aug_matrix[j, i] != 0:
                    aug_matrix[[i, j]] = aug_matrix[[j, i]]  # Swap rows
                    break
            else:
                raise ValueError("No unique solution exists.")

        # Normalize the pivot row
        pivot = aug_matrix[i, i]
        aug_matrix[i] = aug_matrix[i] / pivot

        # Eliminate rows below the pivot
        for j in range(i + 1, n):
            factor = aug_matrix[j, i]
            aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = aug_matrix[i, -1] - np.dot(aug_matrix[i, i + 1:n], x[i + 1:n])

    return x
