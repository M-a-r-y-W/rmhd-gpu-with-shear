from __future__ import annotations

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.testing import single_mode_field
from rmhdgpu.operators import poisson_bracket
from rmhdgpu.workspace import Workspace


def test_workspace_buffers_have_expected_shape_and_dtype() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    workspace = Workspace(grid, backend)

    assert workspace.real["r0"].shape == grid.real_shape
    assert workspace.real["r0"].dtype == grid.real_dtype
    assert workspace.complex["c0"].shape == grid.fourier_shape
    assert workspace.complex["c0"].dtype == grid.complex_dtype


def test_poisson_bracket_can_use_workspace_buffers() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    f_hat = single_mode_field(grid, backend, (1, 0, 1))
    g_hat = single_mode_field(grid, backend, (0, 1, 1))

    result = poisson_bracket(f_hat, g_hat, grid, fft, workspace)

    assert result.shape == grid.fourier_shape
    assert result.dtype == grid.complex_dtype
