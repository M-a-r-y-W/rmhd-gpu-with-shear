from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.random_modes import random_band_limited_field
from rmhdgpu.initconds.testing import single_mode_field
from rmhdgpu.masks import build_dealias_mask


def test_random_band_limited_field_properties() -> None:
    config = Config(Nx=12, Ny=12, Nz=12)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)

    field_hat = random_band_limited_field(
        grid=grid,
        backend=backend,
        fft=fft,
        kmin=1.0,
        kmax=3.0,
        seed=123,
        rms=0.5,
        dealias_mask=mask,
    )
    field_real = backend.to_numpy(fft.c2r(field_hat))
    field_hat_np = backend.to_numpy(field_hat)

    k_mag = np.sqrt(backend.to_numpy(grid.k2))
    band = (k_mag >= 1.0) & (k_mag <= 3.0)
    weights = np.ones(grid.fourier_shape[-1])
    weights[1:-1] = 2.0
    energy_total = np.sum(np.abs(field_hat_np) ** 2 * weights[None, None, :])
    energy_inside = np.sum(np.abs(field_hat_np[band]) ** 2 * np.broadcast_to(weights, grid.fourier_shape)[band])

    assert field_hat.shape == grid.fourier_shape
    assert np.isfinite(field_real).all()
    np.testing.assert_allclose(
        np.sqrt(np.mean(field_real**2)),
        0.5,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert energy_inside / energy_total > 1.0 - 1.0e-12


def test_single_mode_field_properties() -> None:
    config = Config(Nx=8, Ny=8, Nz=8)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)

    field_hat = single_mode_field(grid, backend, mode_indices=(1, 2, 1), amplitude=2.0, phase=0.3)
    field_real = backend.to_numpy(fft.c2r(field_hat))
    field_hat_np = backend.to_numpy(field_hat)

    assert field_hat.shape == grid.fourier_shape
    assert np.isfinite(field_real).all()
    assert np.max(np.abs(field_real)) > 0.0
    assert np.count_nonzero(np.abs(field_hat_np) > 0.0) == 1
    assert np.abs(field_hat_np[1, 2, 1]) > 0.0
