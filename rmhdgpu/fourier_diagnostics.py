"""Generic Fourier-space diagnostic bookkeeping.

The solver stores Fourier fields in `rfftn` layout: all modes are retained in
the first two directions, while the last direction stores only the
non-negative Hermitian half. The helpers here provide the weights and
normalization needed to turn modal densities in that storage format into
real-space volume averages.

Equation modules should define the physical modal density they care about;
this module only handles the generic R2C summation convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_R2C_MODAL_WEIGHTS_CACHE: dict[tuple[str, int, int, int, str], Any] = {}


def r2c_modal_weights(grid: Any, backend: Any) -> Any:
    """Return Hermitian weights for summing an `rfftn` modal density."""

    cache_key = (backend.backend_name, grid.Nx, grid.Ny, grid.Nz, str(grid.real_dtype))
    cached = _R2C_MODAL_WEIGHTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    xp = backend.xp
    weights = xp.ones(grid.fourier_shape, dtype=grid.real_dtype)
    if grid.Nz % 2 == 0:
        weights[..., 1:-1] = 2.0
    else:
        weights[..., 1:] = 2.0
    _R2C_MODAL_WEIGHTS_CACHE[cache_key] = weights
    return weights


def modal_density_average(
    density_hat: Any,
    grid: Any,
    backend: Any,
    *,
    mask: Any | None = None,
) -> float:
    """Return the real-space average represented by an `rfftn` modal density.

    `density_hat` should be the non-negative or signed modal density using the
    package FFT convention: unnormalized forward transforms and inverse
    transforms scaled by `1 / (Nx * Ny * Nz)`. If `mask` is supplied, only modes
    where the mask is nonzero contribute to the average.
    """

    xp = backend.xp
    weighted_density = r2c_modal_weights(grid, backend) * xp.real(density_hat)
    if mask is not None:
        weighted_density = weighted_density * mask
    normalization = float(np.prod(grid.real_shape) ** 2)
    return backend.scalar_to_float(xp.sum(weighted_density) / normalization)


def modal_average(density_hat: Any, grid: Any, backend: Any) -> float:
    """Return the full modal-density average without a mask."""

    return modal_density_average(density_hat, grid, backend)


def modal_inner_product_average(left_hat: Any, right_hat: Any, grid: Any, backend: Any) -> float:
    """Return `<left * right>` for two real fields stored in `rfftn` layout."""

    return modal_average(left_hat * backend.xp.conj(right_hat), grid, backend)
