import numpy as np


def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)

    for i in range(n):
        max_row = np.argmax(abs(U[i:, i])) + i
        if i != max_row:
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            L[[i, max_row], :i] = L[[max_row, i], :i]

        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    return P, L, U

def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Perform QR decomposition using numpy.linalg.qr
    Q, R = np.linalg.qr(x)
    return Q, R


def determinant(x: np.ndarray) -> float:
    if x.shape[0] != x.shape[1]:
        raise ValueError("Determinant can only be calculated for square matrices.")

    return np.linalg.det(x)

def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return eigenvalues, eigenvectors

def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(x, full_matrices=True)
    return U, np.diag(S), Vt