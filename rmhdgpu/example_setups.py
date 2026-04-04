"""Small reusable setup helpers shared by example inputs and visualization scripts."""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.initconds.eigenmodes import alfven_mode_state
from rmhdgpu.operators import dx, dy, lap_perp
from rmhdgpu.state import State


DECAY_LOW_MODE_DEFAULTS = {
    "phi_seed": 1,
    "phi_amplitude": 0.4,
    "psi_seed": 2,
    "psi_amplitude": 0.3,
    "upar_seed": 3,
    "upar_amplitude": 0.08,
    "dbpar_seed": 4,
    "dbpar_amplitude": 0.06,
    "s_seed": 5,
    "s_amplitude": 0.05,
}


def estimate_hyperdiffusion_coefficient(k_d: float, k0: float, u_rms: float, order: int) -> float:
    """Estimate `nu` from `tau_nl^{-1}(k_d) ~ nu * k_d^(2 n)`."""

    return k_d * u_rms * (k_d / k0) ** (-1.0 / 3.0) / (k_d ** (2 * order))


def dealiased_max_kperp(grid: object, backend: object, mask: object) -> float:
    """Return the maximum retained perpendicular wavenumber after dealiasing."""

    kperp = np.sqrt(backend.to_numpy(grid.kperp2))
    retained = backend.to_numpy(mask).astype(bool)
    return float(np.max(kperp[retained]))


def low_mode_real_field(
    grid: object,
    backend: object,
    *,
    seed: int,
    amplitude: float,
) -> object:
    """Return a smooth low-mode random field built from the first few Fourier modes."""

    rng = np.random.default_rng(seed)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = backend.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                a_cos = rng.normal(scale=amplitude / 6.0)
                a_sin = rng.normal(scale=amplitude / 6.0)
                phase = nx * x + ny * y + nz * z
                field += a_cos * xp.cos(phase) + a_sin * xp.sin(phase)

    return field.astype(grid.real_dtype, copy=False)


def aw_packet_real_field(grid: object, backend: object) -> object:
    """Return the real-space field used by the Alfvén-wave packet example."""

    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = xp.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                coefficient = 0.15 / (nx + ny + nz - 1.0)
                phase = 0.3 * (nx - ny + nz)
                field += coefficient * xp.cos(nx * x + ny * y + nz * z + phase)

    rms = backend.scalar_to_float(xp.sqrt(xp.mean(field**2)))
    field *= 1.0 / rms
    return field


def initial_u_rms(phi_hat: object, grid: object, fft: object, backend: object) -> float:
    """Return the perpendicular RMS velocity for a stream-function field."""

    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    xp = backend.xp
    return backend.scalar_to_float(xp.sqrt(xp.mean(ux**2 + uy**2)))


def build_initial_state(
    initial_condition: object,
    *,
    grid: object,
    backend: object,
    fft: object,
    dealias_mask: object | None,
    field_names: list[str],
    params: object,
) -> State:
    """Build a state from one of the small built-in example initial-condition types."""

    mask = dealias_mask
    kind = initial_condition.type
    if kind == "zero":
        return State(grid, backend, field_names=field_names)
    if kind == "alfven_mode":
        return alfven_mode_state(
            grid=grid,
            backend=backend,
            field_names=field_names,
            k_indices=initial_condition.k_indices,
            amplitude=initial_condition.amplitude,
            branch=initial_condition.branch,
            params=params,
        )
    if kind == "aw_packet":
        phi_hat = fft.r2c(aw_packet_real_field(grid, backend))
        if mask is not None:
            phi_hat *= mask
        state = State(grid, backend, field_names=field_names)
        state["psi"][...] = phi_hat
        state["omega"][...] = lap_perp(phi_hat, grid)
        return state
    if kind == "decaying_low_modes":
        defaults = dict(DECAY_LOW_MODE_DEFAULTS)
        defaults.update(dict(initial_condition.parameters))
        state = State(grid, backend, field_names=field_names)
        phi_hat = fft.r2c(
            low_mode_real_field(
                grid,
                backend,
                seed=int(defaults["phi_seed"]),
                amplitude=float(defaults["phi_amplitude"]),
            )
        )
        psi_hat = fft.r2c(
            low_mode_real_field(
                grid,
                backend,
                seed=int(defaults["psi_seed"]),
                amplitude=float(defaults["psi_amplitude"]),
            )
        )
        if mask is not None:
            phi_hat *= mask
            psi_hat *= mask
        state["psi"][...] = psi_hat
        state["omega"][...] = lap_perp(phi_hat, grid)

        for field_name, seed_key, amplitude_key in (
            ("upar", "upar_seed", "upar_amplitude"),
            ("dbpar", "dbpar_seed", "dbpar_amplitude"),
            ("s", "s_seed", "s_amplitude"),
        ):
            field_hat = fft.r2c(
                low_mode_real_field(
                    grid,
                    backend,
                    seed=int(defaults[seed_key]),
                    amplitude=float(defaults[amplitude_key]),
                )
            )
            if mask is not None:
                field_hat *= mask
            state[field_name][...] = field_hat
        return state
    raise ValueError(f"Unsupported initial condition type {kind!r}.")
