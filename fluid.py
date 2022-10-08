import time
import numpy as np
from functools import partial
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized, spsolve
from utils import build_laplacian_matrix, draw_velocity, compute_curl, draw_density, draw_curl, draw_mix


class StableFluids(object):
    def __init__(self, N: int, dt: float, domain: list=[[0, 1], [0, 1]], visc: float=0, diff: float=0, boundary_func=None):

        """Stable Fluids solver with stagger grid discretization.
        Density(dye) and pressure values are stored at the center of grids.
        Horizontal (u) and vertical (v) velocity values are stored at edges.
        The velocity along (i, j) indexing directions are (v, u).
        A layer of boundary is warpped outside.
        TODO: support different pressure solvers
        TODO: support external force
        TODO: support arbitrary rho. now assume rho=1 everywhere.

        ---v-----v---
        |     |     |
        u  d  u  d  u
        |     |     |
        ---v-----v---
        |     |     |
        u  d  u  d  u
        |     |     |
        ---v-----v---

        Args:
            N (int): grid resolution along the longest dimension.
            dt (float): timestep size
            domain (list, optional): 2D domain ([[x_min, x_max], [y_min, y_max]]). 
                Defaults to [[0, 1], [0, 1]].
            visc (float, optional): viscosity coefficient. Defaults to 0.
            diff (float, optional): diffusion coefficient. Defaults to 0.
            boundary_func (function): function to set boundary condition, 
                func(d_grid, u_grid, v_grid) -> None. Defaults to None, using solid boundary.
        """
        len_x = (domain[0][1] - domain[0][0])
        len_y = (domain[1][1] - domain[1][0])
        self.h = max(len_x, len_y) / N
        self.M = int(len_x / self.h)
        self.N = int(len_y / self.h)

        self.dt = dt
        self.visc = visc
        self.diff = diff
        self.domain = domain
        self.timestep = 0
        
        self._d_grid = np.zeros((self.M + 2, self.N + 2))
        self._u_grid = np.zeros((self.M + 2, self.N + 1))
        self._v_grid = np.zeros((self.M + 1, self.N + 2))

        # grid coordinates
        self._grid_indices = np.indices(self._d_grid.shape)

        # interpolation function
        self.interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

        # boundary condition function
        if boundary_func is None: # assume solid boundary by default
            self.boundary_func = self._set_solid_boundary
        else:
            self.boundary_func = boundary_func

        # linear system solver
        print("Build pre-factorized linear system solver. Could take a while.")
        self.lap_mat = build_laplacian_matrix(self.M, self.N) 
        self.pressure_solver = factorized(self.lap_mat)    

        if self.diff > 0:
            self.diffD_solver = factorized(np.identity(self.M * self.N) - 
                diff * dt / self.h / self.h * build_laplacian_matrix(self.M, self.N))
        if self.visc > 0:
            self.diffU_solver = factorized(np.identity(self.M * (self.N + 1)) - 
                visc * dt / self.h / self.h * build_laplacian_matrix(self.M, self.N + 1))
            self.diffV_solver = factorized(np.identity((self.M + 1) * self.N) - 
                visc * dt / self.h / self.h * build_laplacian_matrix(self.M + 1, self.N))

    @property
    def grid_density(self):
        """density values at grid centers"""
        return self._d_grid[1:-1, 1:-1]
    
    @property
    def grid_velocity(self):
        """velocity values at grid centers"""
        u = (self._u_grid[1:-1, 1:] + self._u_grid[1:-1, :-1]) / 2
        v = (self._v_grid[1:, 1:-1] + self._v_grid[:-1, 1:-1]) / 2
        vel = np.stack([v, u], axis=-1)
        return vel
    
    @property
    def grid_curl(self):
        """curl(vorticity) values at grid centers"""
        curl = compute_curl(self.grid_velocity, self.h)
        return curl

    def _transform_coords(self, coords, offset):
        """transform coords in grid space to original domain"""
        minxy = np.array([self.domain[0][0], self.domain[1][0]])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        coords = coords * self.h + minxy
        return coords

    def add_source(self, attr: str, source_func):
        """add source to density(d) or velocity field(u, v)

        Args:
            attr (str): "velocity" or "density"
            source_func (function): attr(x) = source_func(x)

        Raises:
            ValueError: _description_
        """
        if source_func is None:
            return

        if attr == "velocity":
            u_indices = self._transform_coords(self._grid_indices[:, :, :-1], [-0.5, 0])
            self._u_grid += source_func(u_indices)[..., 1]

            v_indices = self._transform_coords(self._grid_indices[:, :-1], [0, -0.5])
            self._v_grid += source_func(v_indices)[..., 0]
        elif attr == "density":
            d_indices = self._transform_coords(self._grid_indices, [-0.5, -0.5])
            self._d_grid += source_func(d_indices)
        else:
            raise ValueError(f"attr must be velocity or density, but got {attr}.")
        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

    def step(self):
        """Integrates the system forward in time by dt."""
        since = time.time()
        
        self._velocity_step()
        self._density_step()
        self.timestep += 1
        
        timecost = time.time() - since
        return timecost
    
    def _density_step(self):
        """update density field by one timestep"""
        # diffusion
        if self.diff > 0:
            self._diffuseD()

        # advection
        self._advectD()

    def _velocity_step(self):
        """update density field by one timestep"""
        # external force
        pass

        # advection
        self._advectVel()

        # diffusion
        if self.visc > 0:
            self._diffuseU()
            self._diffuseV()

        # projection
        self._project()

    def _diffuseD(self):
        """diffusion step for d ([1, M], [1, N]) using implicit method"""
        self._d_grid[1:-1, 1:-1] = self.diffD_solver(self._d_grid[1:-1, 1:-1].flatten()).reshape(self.M, self.N)

    def _diffuseU(self):
        """diffusion step for u ([1, M], [0, N]) using implicit method"""
        self._u_grid[1:-1, :] = self.diffU_solver(self._u_grid[1:-1].flatten()).reshape(self.M, self.N + 1)

    def _diffuseV(self):
        """diffusion step for v ([0, M], [1, N]) using implicit method"""
        self._v_grid[:, 1:-1] = self.diffV_solver(self._v_grid[:, 1:-1].flatten()).reshape(self.M + 1, self.N)

    def _advectD(self):
        """advect density for ([1, M], [1, N])"""
        i_back = self._grid_indices[0, 1:-1, 1:-1] - self.dt / self.h * (
            self._v_grid[:-1, 1:-1] + self._v_grid[1:, 1:-1]) / 2
        j_back = self._grid_indices[1, 1:-1, 1:-1] - self.dt / self.h * (
            self._u_grid[1:-1, :-1] + self._u_grid[1:-1, 1:]) / 2
        self._d_grid[1:-1, 1:-1] = self.interpolate(self._d_grid, np.stack([i_back, j_back]))

    def _advectVel(self):
        """addvect velocity field"""
        new_u_grid = self._advectU()
        new_v_grid = self._advectV()
        self._u_grid[1:-1, :] = new_u_grid
        self._v_grid[:, 1:-1] = new_v_grid

    def _advectU(self):
        """advect horizontal velocity (u) for ([1, M], [0, N])"""
        i_back = self._grid_indices[0, 1:-1, :-1] - self.dt / self.h * (
            self._v_grid[:-1, :-1] + self._v_grid[1:, :-1] + self._v_grid[:-1, 1:] + self._v_grid[1:, 1:]) / 4
        j_back = self._grid_indices[1, 1:-1, :-1] - self.dt / self.h * self._u_grid[1:-1]
        i_back = np.clip(i_back, 0.5, self.M + 0.5)
        new_u_grid = self.interpolate(self._u_grid, np.stack([i_back, j_back]))
        return new_u_grid

    def _advectV(self):
        """advect vertical velocity (v) for ([0, M], [1, N])"""
        i_back = self._grid_indices[0, :-1, 1:-1] - self.dt / self.h * self._v_grid[:, 1:-1]
        j_back = self._grid_indices[1, :-1, 1:-1] - self.dt / self.h * (
            self._u_grid[:-1, :-1] + self._u_grid[1:, :-1] + self._u_grid[:-1, 1:] + self._u_grid[1:, 1:]) / 4
        j_back = np.clip(j_back, 0.5, self.N + 0.5)
        new_v_grid = self.interpolate(self._v_grid, np.stack([i_back, j_back]))
        return new_v_grid

    def _solve_pressure(self):
        """solve pressure field for laplacian(pressure) = divergence(velocity)"""
        # compute divergence of velocity field
        h = self.h
        div = (self._u_grid[1:-1, 1:] - self._u_grid[1:-1, :-1] + 
            self._v_grid[1:, 1:-1] - self._v_grid[:-1, 1:-1]) / h
        
        # solve poisson equation as a linear system
        rhs = div.flatten() * h * h
        p_grid = self.pressure_solver(rhs).reshape(self.M, self.N)
        return p_grid

    def _project(self):
        """projection step to enforce divergence free (incompressible flow)"""
        # set solid boundary condition
        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

        p_grid = self._solve_pressure()

        # apply gradient of pressure to correct velocity field
        # ([1, M], [1, N-1]) for u, ([1, M-1], [1, N]) for v. FIXME: this range holds only for solid boundary.
        self._u_grid[1:-1, 1:-1] -= (p_grid[:, 1:] - p_grid[:, :-1]) / self.h
        self._v_grid[1:-1, 1:-1] -= (p_grid[1:, :] - p_grid[:-1, :]) / self.h

        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

    def _set_solid_boundary(self, d_grid, u_grid, v_grid, h):
        """set solid (zero Dirichlet) boundary condition"""
        for grid in [d_grid, u_grid, v_grid]:
            grid[0, :] = 0
            grid[-1, :] = 0
            grid[:, 0] = 0
            grid[:, -1] = 0

    def draw(self, attr: str, save_path: str):
        """draw a frame"""
        since = time.time()

        if attr == "velocity":
            draw_velocity(self.grid_velocity, save_path)
        elif attr == "curl":
            draw_curl(self.grid_curl, save_path)
        elif attr == "density":
            draw_density(self.grid_density, save_path)
        elif attr == "mix":
            draw_mix(self.grid_curl, self.grid_density, save_path)
        else:
            raise NotImplementedError

        timecost = time.time() - since
        return timecost
