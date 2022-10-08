import numpy as np


def taylorgreen_velocity(coords: np.ndarray):
    """taylorgreen velocity. domain: [0, 2pi] x [0, 2pi].

    Args:
        coords (np.ndarray): coordinates of shape (..., 2)

    Returns:
        velocity: velocity at coordinates
    """
    A = 1
    a = 1
    B = -1
    b = 1
    x = coords[..., 0]
    y = coords[..., 1]
    u = A * np.sin(a * x) * np.cos(b * y)
    v = B * np.cos(a * x) * np.sin(b * y)
    vel = np.stack([u, v], axis=-1)
    return vel


setup = {
    "domain": [[0, 2 * np.pi], [0, 2 * np.pi]],
    "vsource": taylorgreen_velocity,
    "dsource": None,
    "src_duration": 1
}
