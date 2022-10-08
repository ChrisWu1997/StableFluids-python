# from https://github.com/GregTJ/stable-fluids/blob/main/example.py
import numpy as np


def create_circular_points(point_dist: float, point_count: int):
    points = np.linspace(-np.pi, np.pi, point_count, endpoint=False)
    points = tuple(np.array((np.cos(p), np.sin(p))) for p in points)
    normals = tuple(-p for p in points)
    points = tuple(point_dist * p for p in points)
    return points, normals


def inflow_velocity(coords: np.ndarray, vel: float=0.01, point_radius: float=0.05, point_dist: float=0.8, point_count: int=4):
    points, normals = create_circular_points(point_dist, point_count)

    inflow_velocity = np.zeros_like(coords)
    for p, n in zip(points, normals):
        mask = np.linalg.norm(coords - p[None, None], 2, axis=-1) <= point_radius
        inflow_velocity[mask] += n[None] * vel
    return inflow_velocity


def inflow_density(coords: np.ndarray, point_radius: float=0.05, point_dist: float=0.8, point_count: int=4):
    points, normals = create_circular_points(point_dist, point_count)

    inflow_density = np.zeros(coords.shape[:-1])
    for p in points:
        mask = np.linalg.norm(coords - p[None, None], 2, axis=-1) <= point_radius
        inflow_density[mask] = 1
    return inflow_density


setup = {
    "domain": [[-1, 1], [-1, 1]],
    "vsource": inflow_velocity,
    "dsource": inflow_density,
    "src_duration": 60,
    "boundary_func": None
}
