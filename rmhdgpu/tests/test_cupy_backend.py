from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


cupy = pytest.importorskip("cupy")
try:
    cupy.zeros((1,), dtype=cupy.float64)
except Exception as exc:  # pragma: no cover - depends on runtime availability
    pytest.skip(f"CuPy is installed but not usable in this environment: {exc}", allow_module_level=True)


def _build_context() -> tuple[Config, object, object, FFTManager]:
    config = Config(Nx=8, Ny=8, Nz=8, backend="cupy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft


def test_build_cupy_backend() -> None:
    config = Config(backend="cupy")
    backend = build_backend(config)

    assert backend.xp is cupy
    assert backend.backend_name == "cupy"
    assert backend.is_gpu is True


def test_cupy_fft_roundtrip() -> None:
    _, backend, grid, fft = _build_context()
    field = cupy.random.default_rng(123).standard_normal(grid.real_shape, dtype=grid.real_dtype)

    recovered = fft.c2r(fft.r2c(field))

    np.testing.assert_allclose(
        cupy.asnumpy(recovered),
        cupy.asnumpy(field),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_workspace_allocates_on_device() -> None:
    config, backend, grid, _ = _build_context()
    workspace = Workspace(grid, backend)

    assert isinstance(workspace.real["r0"], cupy.ndarray)
    assert isinstance(workspace.complex["c0"], cupy.ndarray)
    assert workspace.real["r0"].dtype == config.real_dtype
    assert workspace.complex["c0"].dtype == config.complex_dtype


def test_state_allocates_on_device() -> None:
    _, backend, grid, _ = _build_context()
    state = State(grid, backend, field_names=["psi", "omega"])

    for name in state.field_names:
        assert isinstance(state[name], cupy.ndarray)
        assert state[name].shape == grid.fourier_shape
