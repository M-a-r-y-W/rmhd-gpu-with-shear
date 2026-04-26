from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.initconds.testing import random_spectrum_test_parameters, single_mode_field
from rmhdgpu.masks import build_dealias_mask


def test_random_spectrum_builder_returns_finite_nonzero_state() -> None:
    config = Config(Nx=12, Ny=12, Nz=12)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)

    state = build_initial_state(
        "random_spectrum",
        parameters=random_spectrum_test_parameters(0.5),
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    for name in config.field_names:
        field_hat = backend.to_numpy(state[name])
        field_real = backend.to_numpy(fft.c2r(state[name]))
        assert field_hat.shape == grid.fourier_shape
        assert np.isfinite(field_hat).all()
        assert np.isfinite(field_real).all()
        assert np.max(np.abs(field_real)) > 0.0
    np.testing.assert_allclose(s09.total_energy(state, grid, backend, config), 0.5, atol=1.0e-12, rtol=1.0e-12)


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
