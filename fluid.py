import numpy as np
from functools import partial
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized, spsolve
from utils import build_laplacian_matrix, draw_velocity, compute_curl, draw_density, draw_curl, draw_mix


class StableFluids(object):
    def __init__(self, N: int, dt: float, domain: list=[[0, 1], [0, 1]], visc: float=0, diff: float=0):

        """Stable Fluids solver with stagger grid discretization.
        Density(dye) and pressure values are stored at the center of grids.
        Horizontal (u) and vertical (v) velocity values are stored at edges.
        The velocity along (i, j) indexing directions are (v, u).
        A layer of boundary is warpped outside.
        TODO: support arbitrary M/N and arbitrary domains
        TODO: support different pressure solvers
        TODO: support different boundary conditions
        TODO: support external force

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
            N (int): grid resolution, discretize domain [0, 1] x [0, 1].
            dt (float): timestep size
            domain (list, optional): 2D domain ([[x_min, x_max], [y_min, y_max]]). 
                Defaults to [[0, 1], [0, 1]].
            visc (float, optional): viscosity coefficient. Defaults to 0.
            diff (float, optional): diffusion coefficient. Defaults to 0.
        """
        self.N = N
        self.dt = dt
        self.visc = visc
        self.diff = diff
        self.domain = domain
        self.timestep = 0
        
        self._d_grid = np.zeros((N + 2, N + 2))
        self._u_grid = np.zeros((N + 2, N + 1))
        self._v_grid = np.zeros((N + 1, N + 2))
        self._grid_coords = np.indices(self._d_grid.shape)
        self.h = (domain[0][1] - domain[0][0]) / N # size of each grid

        # interpolation function
        self.interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

        # linear system solver
        self.lap_mat = build_laplacian_matrix(N, N) 
        self.pressure_solver = factorized(self.lap_mat)    

        if self.diff > 0:
            self.diffD_solver = factorized(np.identity(N * N) - 
                diff * dt * N * N * build_laplacian_matrix(N, N))
        if self.visc > 0:
            self.diffU_solver = factorized(np.identity(N * (N + 1)) - 
                visc * dt * N * N * build_laplacian_matrix(N, N + 1))
            self.diffV_solver = factorized(np.identity((N + 1) * N) - 
                visc * dt * N * N * build_laplacian_matrix(N + 1, N))

    @property
    def grid_density(self):
        """density values at grid centers"""
        return self._d_grid[1:-1, 1:-1].copy()
    
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
            u_indices = self._transform_coords(self._grid_coords[:, :, :-1], [-0.5, 0])
            self._u_grid += source_func(u_indices)[..., 1]

            v_indices = self._transform_coords(self._grid_coords[:, :-1], [0, -0.5])
            self._v_grid += source_func(v_indices)[..., 0]
        elif attr == "density":
            d_indices = self._transform_coords(self._grid_coords, [-0.5, -0.5])
            self._d_grid += source_func(d_indices)
        else:
            raise ValueError(f"attr must be velocity or density, but got {attr}.")
        self._set_solid_boundary()

    def step(self):
        """Integrates the system forward in time by dt."""
        self._velocity_step()
        self._density_step()
        self.timestep += 1
    
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
        """diffusion step for d ([1, N], [1, N]) using implicit method"""
        self._d_grid[1:-1, 1:-1] = self.diffD_solver(self._d_grid[1:-1, 1:-1].flatten()).reshape(self.N, self.N)

    def _diffuseU(self):
        """diffusion step for u ([1, N], [0, N]) using implicit method"""
        self._u_grid[1:-1, :] = self.diffU_solver(self._u_grid[1:-1].flatten()).reshape(self.N, self.N + 1)

    def _diffuseV(self):
        """diffusion step for v ([0, N], [1, N]) using implicit method"""
        self._v_grid[:, 1:-1] = self.diffV_solver(self._v_grid[:, 1:-1].flatten()).reshape(self.N + 1, self.N)

    def _advectD(self):
        """advect density for ([1, N], [1, N])"""
        i_back = self._grid_coords[0, 1:-1, 1:-1] - self.dt * self.N * (
            self._v_grid[:-1, 1:-1] + self._v_grid[1:, 1:-1]) / 2
        j_back = self._grid_coords[1, 1:-1, 1:-1] - self.dt * self.N * (
            self._u_grid[1:-1, :-1] + self._u_grid[1:-1, 1:]) / 2
        self._d_grid[1:-1, 1:-1] = self.interpolate(self._d_grid, np.stack([i_back, j_back]))

    def _advectVel(self):
        """addvect velocity field"""
        new_u_grid = self._advectU()
        new_v_grid = self._advectV()
        self._u_grid[1:-1, :] = new_u_grid
        self._v_grid[:, 1:-1] = new_v_grid

    def _advectU(self):
        """advect horizontal velocity (u) for ([1, N], [0, N])"""
        i_back = self._grid_coords[0, 1:-1, :-1] - self.dt / self.h * (
            self._v_grid[:-1, :-1] + self._v_grid[1:, :-1] + self._v_grid[:-1, 1:] + self._v_grid[1:, 1:]) / 4
        j_back = self._grid_coords[1, 1:-1, :-1] - self.dt / self.h * self._u_grid[1:-1]
        i_back = np.clip(i_back, 0.5, self.N + 0.5)
        new_u_grid = self.interpolate(self._u_grid, np.stack([i_back, j_back]))
        return new_u_grid

    def _advectV(self):
        """advect vertical velocity (v) for ([0, N], [1, N])"""
        i_back = self._grid_coords[0, :-1, 1:-1] - self.dt / self.h * self._v_grid[:, 1:-1]
        j_back = self._grid_coords[1, :-1, 1:-1] - self.dt / self.h * (
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
        p_grid = self.pressure_solver(rhs)
        p_grid = p_grid.reshape(self.N, self.N) # (N, N)
        return p_grid

    def _project(self):
        """projection step to enforce divergence free (incompressible flow)"""
        # set solid boundary condition
        self._set_solid_boundary()

        p_grid = self._solve_pressure()

        # apply gradient of pressure to correct velocity field
        # ([1, N], [1, N-1]) for u, ([1, N-1], [1, N]) for v
        self._u_grid[1:-1, 1:-1] -= (p_grid[:, 1:] - p_grid[:, :-1]) / self.h
        self._v_grid[1:-1, 1:-1] -= (p_grid[1:, :] - p_grid[:-1, :]) / self.h

        self._set_solid_boundary()

    def _set_solid_boundary(self):
        """set solid (zero) boundary condition"""
        self._d_grid[0, :] = 0
        self._d_grid[-1, :] = 0
        self._d_grid[:, 0] = 0
        self._d_grid[:, -1] = 0
        self._u_grid[0, :] = 0
        self._u_grid[-1, :] = 0
        self._u_grid[:, 0] = 0
        self._u_grid[:, -1] = 0
        self._v_grid[0, :] = 0
        self._v_grid[-1, :] = 0
        self._v_grid[:, 0] = 0
        self._v_grid[:, -1] = 0

    def draw(self, attr: str, save_path: str):
        """draw a frame"""
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
