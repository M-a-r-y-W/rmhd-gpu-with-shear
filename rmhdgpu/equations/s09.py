"""Ideal homogeneous Schekochihin-2009-style five-field prototype.

This module implements the nondissipative homogeneous system in Fourier space
using the package-wide conventions:

- `z` is the parallel direction
- real arrays have shape `(Nx, Ny, Nz)`
- Fourier arrays have shape `(Nx, Ny, Nz//2 + 1)` from `rfftn`
- `lap_perp(f_hat) = -k_perp^2 f_hat`
- `inv_lap_perp(f_hat) = -inv_kperp2 f_hat`

The evolved fields are `[psi, omega, upar, dbpar, s]`, with the derived
potential `phi = inv_lap_perp(omega)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rmhdgpu.operators import dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.state import State


FIELD_NAMES = ["psi", "omega", "upar", "dbpar", "s"]


def _get_param(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def alpha_from_params(params: Any) -> float:
    """Return `alpha = chi / (1 + chi)` from the supplied parameter object."""

    chi = _get_param(params, "cs2_over_vA2")
    return chi / (1.0 + chi)


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`.

    The `k_perp = 0` modes are regularized to zero through the grid's
    `inv_kperp2` definition, so `phi_hat` is only meaningful in the RMHD
    subspace with nonzero perpendicular wavenumber.
    """

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def _field_dissipation_spec(params: Any, field_name: str) -> Mapping[str, float | int]:
    if isinstance(params, Mapping):
        return params["dissipation"][field_name]
    return getattr(params, "dissipation")[field_name]


def dissipation_operator(grid: Any, params: Any, field_name: str) -> Any:
    """Return the nonnegative diagonal damping operator `D_i(k)` for one field."""

    spec = _field_dissipation_spec(params, field_name)
    nu_perp = float(spec["nu_perp"])
    nu_par = float(spec["nu_par"])
    n_perp = int(spec["n_perp"])
    n_par = int(spec["n_par"])

    operator = 0.0
    if nu_perp > 0.0:
        operator = operator + nu_perp * (grid.kperp2**n_perp)
    if nu_par > 0.0:
        operator = operator + nu_par * (grid.kpar2**n_par)
    if isinstance(operator, float):
        operator = grid.kperp2 * 0.0
    return operator


def build_dissipation_operators(
    grid: Any,
    params: Any,
    field_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the diagonal damping operators for all evolved fields."""

    names = FIELD_NAMES if field_names is None else field_names
    return {name: dissipation_operator(grid, params, name) for name in names}


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space RHS of the ideal homogeneous five-field system."""

    vA = _get_param(params, "vA")
    alpha = alpha_from_params(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    upar_hat = state["upar"]
    dbpar_hat = state["dbpar"]
    s_hat = state["s"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    rhs_psi = rhs_state["psi"]
    rhs_psi[...] = vA * dz(phi_hat, grid)
    rhs_psi[...] -= poisson_bracket(
        phi_hat,
        psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_omega = rhs_state["omega"]
    rhs_omega[...] = vA * dz(lap_psi_hat, grid)
    rhs_omega[...] -= poisson_bracket(
        phi_hat,
        omega_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_omega[...] += poisson_bracket(
        psi_hat,
        lap_psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_dbpar = rhs_state["dbpar"]
    rhs_dbpar[...] = alpha * dz(upar_hat, grid)
    rhs_dbpar[...] -= poisson_bracket(
        phi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] += alpha * poisson_bracket(
        psi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_upar = rhs_state["upar"]
    rhs_upar[...] = (vA**2) * dz(dbpar_hat, grid)
    rhs_upar[...] -= poisson_bracket(
        phi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_upar[...] += (vA**2) * poisson_bracket(
        psi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_s = rhs_state["s"]
    rhs_s[...] = 0.0
    rhs_s[...] -= poisson_bracket(
        phi_hat,
        s_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    return rhs_state


def rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Backward-compatible alias for the ideal RHS.

    Dissipation is intentionally not included here. The integrating-factor
    timestepper applies the diagonal damping operators separately so the ideal
    system remains directly accessible for invariant and linear-behavior tests.
    """

    return ideal_rhs(state, grid, fft, workspace, params, dealias_mask=dealias_mask, out=out)


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 5x5 linear matrix for one Fourier mode.

    The field order is `[psi, omega, upar, dbpar, s]`. For `k_perp = 0`, the
    `psi/omega` Alfvénic block is set to zero because the inverse perpendicular
    Laplacian is not meaningful there in the RMHD subspace.
    """

    vA = _get_param(params, "vA")
    alpha = alpha_from_params(params)
    matrix = np.zeros((5, 5), dtype=np.complex128)
    ikz = 1j * float(kz)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -vA * ikz / kperp2
        matrix[1, 0] = -vA * ikz * kperp2

    matrix[2, 3] = (vA**2) * ikz
    matrix[3, 2] = alpha * ikz
    return matrix
