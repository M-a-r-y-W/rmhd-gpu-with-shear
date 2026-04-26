"""Small testing-oriented helpers for verification code."""

from __future__ import annotations

from typing import Any

import numpy as np


def single_mode_field(
    grid: Any,
    backend: Any,
    mode_indices: tuple[int, int, int],
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Any:
    """Return a Fourier field with one stored `rfftn` mode populated."""

    nx, ny, nz = mode_indices
    if nz < 0 or nz > (grid.Nz // 2):
        raise ValueError(
            f"n_z must satisfy 0 <= n_z <= Nz//2; got n_z={nz} for Nz={grid.Nz}."
        )

    field_hat = backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
    coeff = amplitude * np.exp(1j * phase)

    ix = nx % grid.Nx
    iy = ny % grid.Ny
    field_hat[ix, iy, nz] = coeff
    return field_hat


def random_spectrum_test_parameters(init_energy: float = 0.1) -> dict[str, float | int]:
    """Return deterministic random-spectrum parameters for test-sized states."""

    return {
        "n_min": 1.0,
        "n_max": 3.0,
        "alpha": 0.0,
        "init_energy": init_energy,
        "seed": 1,
    }
