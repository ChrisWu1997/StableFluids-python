import numpy as np
from scipy import sparse


def build_laplacian_matrix(M: int, N: int):
    """build laplacian matrix with neumann boundary condition,
    i.e., gradient at boundary equals zero.
    This eliminates one degree of freedom.

    Args:
        M (int): number of rows
        N (int): number of cols

    Returns:
        np.array: laplacian matrix 
    """
    main_diag = np.full(M * N, -4)
    main_diag[[0, N - 1, -N, -1]] = -2
    main_diag[[*range(1, N - 1), *range(-N + 1, -1), 
               *range(N, (M - 2) * N + 1, N), *range(2 * N - 1, (M - 1) * N, N)]] = -3
    side_diag = np.ones(M * N - 1)
    side_diag[[*range(N - 1, M * N - 1, N)]] = 0
    data = [np.ones(M * N - N), side_diag, main_diag, side_diag, np.ones(M * N - N)]
    offsets = [-N, -1, 0, 1, N]
    mat = sparse.diags(data, offsets)
    return mat


def compute_curl(velocity_field: np.ndarray, h: float):
    """compute curl(vorticity) of a 2D velocity field

    Args:
        velocity_field (np.ndarray): velocity field of shape (M, N, 2)
        h (float): grid size
    """
    dvy_dx = np.gradient(velocity_field[..., 1], h)[0]
    dvx_dy = np.gradient(velocity_field[..., 0], h)[1]
    curl = dvy_dx - dvx_dy
    return curl
