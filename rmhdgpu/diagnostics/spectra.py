"""Simple perpendicular shell spectra."""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.equations.s09 import derive_phi_hat


PERPENDICULAR_SPECTRUM_KEYS = ("u_perp", "b_perp", "upar", "dbpar", "s")
_SPECTRUM_GRID_CACHE: dict[tuple[int, int, int, float, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _rfft_weights(grid: Any) -> np.ndarray:
    weights = np.ones(grid.fourier_shape, dtype=np.float64)
    if grid.Nz % 2 == 0:
        weights[..., 1:-1] = 2.0
    else:
        weights[..., 1:] = 2.0
    return weights


def _cached_spectrum_grid_arrays(grid: Any, backend: Any) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (grid.Nx, grid.Ny, grid.Nz, float(grid.Lx), float(grid.Ly), float(grid.Lz))
    cached = _SPECTRUM_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kperp_np = np.sqrt(backend.to_numpy(grid.kperp2))
    weights = _rfft_weights(grid)
    _SPECTRUM_GRID_CACHE[cache_key] = (kperp_np, weights)
    return kperp_np, weights


def perpendicular_shell_spectrum(
    density_hat: Any,
    grid: Any,
    backend: Any,
    bin_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a Fourier-space modal density into perpendicular shells.

    The shell spectrum is the sum of the supplied modal density over shells in
    `k_perp = sqrt(kx^2 + ky^2)`, with the omitted negative-`kz` half of the
    real FFT accounted for by the standard one-sided rFFT weights.

    The returned normalization is a volume average: shell values sum to the
    total weighted modal density divided by `N^2`, where
    `N = Nx * Ny * Nz`.
    """

    density_np = backend.to_numpy(density_hat)
    kperp_np, weights = _cached_spectrum_grid_arrays(grid, backend)
    normalization = float(np.prod(grid.real_shape) ** 2)
    modal_density = weights * density_np / normalization

    if bin_width is None:
        bin_width = min(2.0 * np.pi / grid.Lx, 2.0 * np.pi / grid.Ly)

    max_kperp = float(kperp_np.max())
    edges = np.arange(0.0, max_kperp + 1.5 * bin_width, bin_width)
    spectrum = np.zeros(edges.size - 1, dtype=np.float64)

    shell_index = np.floor(kperp_np.ravel() / bin_width).astype(int)
    flat_density = modal_density.ravel()
    valid = (shell_index >= 0) & (shell_index < spectrum.size)
    np.add.at(spectrum, shell_index[valid], flat_density[valid])

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, spectrum


def perpendicular_energy_spectrum_from_state(
    state: Any,
    grid: Any,
    backend: Any | None = None,
    bin_width: float | None = None,
) -> dict[str, np.ndarray]:
    """Return simple perpendicular shell spectra for the five-field system."""

    backend_obj = state.backend if backend is None else backend
    phi_hat = derive_phi_hat(state["omega"], grid)
    kperp2 = grid.kperp2
    xp = backend_obj.xp

    kperp, u_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(phi_hat) ** 2),
        grid,
        backend_obj,
        bin_width=bin_width,
    )
    _, b_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(state["psi"]) ** 2),
        grid,
        backend_obj,
        bin_width=bin_width,
    )
    _, upar = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["upar"]) ** 2),
        grid,
        backend_obj,
        bin_width=bin_width,
    )
    _, dbpar = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["dbpar"]) ** 2),
        grid,
        backend_obj,
        bin_width=bin_width,
    )
    _, entropy = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["s"]) ** 2),
        grid,
        backend_obj,
        bin_width=bin_width,
    )
    return {
        "kperp": kperp,
        "u_perp": u_perp,
        "b_perp": b_perp,
        "upar": upar,
        "dbpar": dbpar,
        "s": entropy,
    }


def compute_placeholder_spectra(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return an empty placeholder spectral diagnostics payload."""

    return {}
