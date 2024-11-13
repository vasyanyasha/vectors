import numpy as np
def negative_matrix(x: np.ndarray) -> np.ndarray:
    return -1 * x

def reverse_matrix(x: np.ndarray) -> np.ndarray:
    return x[::-1, ::-1]

def affine_transform(
        x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
        translate: tuple[float, float]
) -> np.ndarray:
    if x.shape[0] != 2:
        raise ValueError("Input must be a 2D vector or set of 2D points.")

    alpha_rad = np.radians(alpha_deg)

    rotation_matrix = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad)],
        [np.sin(alpha_rad), np.cos(alpha_rad)]
    ])

    scaling_matrix = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])

    shear_matrix = np.array([
        [1, shear[0]],
        [shear[1], 1]
    ])

    # Combining transformations: rotation -> scale -> shear
    transformation_matrix = np.eye(3)
    transformation_matrix[:2, :2] = rotation_matrix @ scaling_matrix @ shear_matrix

    # Translation
    transformation_matrix[:2, 2] = translate

    # Applying the transformation to the input vector or matrix
    homogeneous_x = np.vstack([x, np.ones((1, x.shape[1]))])
    transformed_x = transformation_matrix @ homogeneous_x

    return transformed_x[:2, :]