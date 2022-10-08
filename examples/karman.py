import numpy as np


KARMAN_VEL = 0.5

def karman_velocity(coords: np.ndarray, init_vel: float=KARMAN_VEL):
    vel = np.zeros_like(coords)
    vel[..., 1] = init_vel # constant horizontal velocity
    return vel


def sphere_obstacle(coords: np.ndarray):
    center = np.array([np.pi / 2, np.pi / 4])
    radius = np.pi / 15
    center = center.reshape(*[1]*len(coords.shape[:-1]), 2)
    sign_dist = np.linalg.norm(coords - center, 2, axis=-1) - radius
    return sign_dist


def karman_boundary(d_grid, u_grid, v_grid, h):
    """d, u, v from stagger grid. h: grid size"""
    grid_indices = np.indices(d_grid.shape)

    def _transform_coords(coords, offset):
        """transform coords in grid space to original domain"""
        minxy = np.array([0, 0])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        coords = coords * h + minxy
        return coords

    # solid sphere obstacle
    u_coords = _transform_coords(grid_indices[:, :, :-1], [-0.5, 0])
    v_coords = _transform_coords(grid_indices[:, :-1], [0, -0.5])
    mask_u = sphere_obstacle(u_coords) < 0
    mask_v = sphere_obstacle(v_coords) < 0
    u_grid[mask_u] = 0
    v_grid[mask_v] = 0

    # domain boundary: open right side
    u_grid[0, :] = 0
    u_grid[-1, :] = 0
    u_grid[:, 0] = KARMAN_VEL

    v_grid[0, :] = 0
    v_grid[-1, :] = 0
    v_grid[:, 0] = 0
    v_grid[:, -1] = 0

    d_grid[0, :] = 0
    d_grid[-1, :] = 0
    d_grid[:, 0] = 0
    d_grid[:, -1] = 0


setup = {
    "domain": [[0, np.pi], [0, 2 * np.pi]],
    "vsource": karman_velocity,
    "dsource": None,
    "src_duration": 1,
    "boundary_func": karman_boundary
}
