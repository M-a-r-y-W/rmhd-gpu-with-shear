from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.fourier_diagnostics import modal_average, modal_density_average, r2c_modal_weights
from rmhdgpu.grid import build_grid


def _build_context() -> tuple[object, object, FFTManager]:
    config = Config(Nx=6, Ny=4, Nz=8, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return backend, grid, fft


def test_modal_average_matches_real_space_parseval() -> None:
    backend, grid, fft = _build_context()
    rng = np.random.default_rng(123)
    field = rng.normal(size=grid.real_shape)
    field_hat = fft.r2c(field)

    modal_value = modal_average(0.5 * np.abs(field_hat) ** 2, grid, backend)
    real_value = float(0.5 * np.mean(field**2))

    np.testing.assert_allclose(modal_value, real_value, atol=1.0e-14, rtol=1.0e-14)


def test_modal_density_average_applies_r2c_weights_and_mask() -> None:
    backend, grid, _ = _build_context()
    density_hat = np.ones(grid.fourier_shape)
    mask = np.zeros(grid.fourier_shape)
    mask[..., 1] = 1.0

    normalization = float(np.prod(grid.real_shape) ** 2)
    expected = float(np.sum(r2c_modal_weights(grid, backend) * mask) / normalization)

    np.testing.assert_allclose(
        modal_density_average(density_hat, grid, backend, mask=mask),
        expected,
        atol=0.0,
        rtol=0.0,
    )
