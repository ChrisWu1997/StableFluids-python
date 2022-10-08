import numpy as np


def vortexsheet_velocity(coords: np.ndarray, rigidR=0.5, rate=0.1):
    w = 1 * 1.0 / rate
    R = np.linalg.norm(coords, 2, axis=-1)
    mask = R < rigidR
    weight = 1
    u = w * coords[..., 1] * weight
    v = -w * coords[..., 0] * weight
    u[~mask] = 0
    v[~mask] = 0
    vel = np.stack([u, v], axis=-1)
    return vel


def vortexsheet_density(coords: np.ndarray, rigidR=0.5):
    R = np.linalg.norm(coords, 2, axis=-1)
    den = np.zeros(coords.shape[:-1])
    den[R < rigidR] = 1.0
    return den


setup = {
    "domain": [[-1, 1], [-1, 1]],
    "vsource": vortexsheet_velocity,
    "dsource": vortexsheet_density,
    "src_duration": 1,
    "boundary_func": None
}
