"""Homogeneous four-field equation set.

This module is intended to be the main physics-facing file for this equation
set. Generic solver bookkeeping should live elsewhere; the functions here
define the evolved fields, derived parameters, derived fields, ideal RHS,
linear representation, dissipation operators, energy, and budget terms.

Fourier conventions used throughout the solver:

- `z` is the parallel direction
- real arrays have shape `(Nx, Ny, Nz)`
- Fourier arrays have shape `(Nx, Ny, Nz//2 + 1)` from `rfftn`
- `lap_perp(f_hat) = -k_perp^2 f_hat`
- `inv_lap_perp(f_hat) = -inv_kperp2 f_hat`

The evolved fields are `[psi, omega, upar, dbpar]`, with
`phi = inv_lap_perp(omega)`. The homogeneous ideal equations are

- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi}`
- `dbpar_t = alpha * dz(upar) - {phi, dbpar} + alpha * {psi, upar}`
- `upar_t = vA^2 * dz(dbpar) - {phi, upar} + vA^2 * {psi, dbpar}`

with `chi = cs2_over_vA2 = cs^2 / vA^2` and `alpha = chi / (1 + chi)`.
"""
# EDIT THIS AFTERWARDS

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.fourier_diagnostics import modal_average, modal_inner_product_average
from rmhdgpu.operators import dy, dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.diagnostics.spectra import perpendicular_shell_spectrum
from rmhdgpu.state import State


EQUATION_SET_NAME = "shear"
FIELD_NAMES = ["psi", "omega", "upar", "dbpar"] 
DEFAULT_INITIAL_CONDITION = "alfven_mode"

# Scalar diagnostic names provided by this equation module. The standard
# `total_energy*` names are what budget plotting tools expect. Additional
# entries are S09-specific energy partitions useful for quick run inspection.
SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "alfvenic_energy": "Alfvenic part of the S09 energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
    "upar_energy": "Unweighted kinetic parallel energy proxy: 0.5 <upar^2>.",
    "dbpar_energy": "Unweighted magnetic-compressive energy proxy: 0.5 <dbpar^2>.",
    "total_energy_proxy": "Legacy unweighted sum of alfvenic_energy, upar_energy, dbpar_energy.",
    "total_energy_rhs_shear": "Signed shear contribution to d_t total_energy.",
}


@dataclass(frozen=True, slots=True)
class S09Parameters:
    """Scalar parameters and diagnostic weights used by this equation set."""

    vA: float
    chi: float
    alpha: float
    Ku: float # Introduce changes here
    dbpar_energy_weight: float


def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> S09Parameters:
    """Return the compact set of scalars used by the S09 equations.

    This is the first place to edit if a new parameter enters the physics.
    """

    vA = _param_float(params, "vA")
    chi = _param_float(params, "cs2_over_vA2")
    Ku = _param_float(params, "Ku") #Introduce changes here
    alpha = chi / (1.0 + chi)
    return S09Parameters(
        vA=vA,
        chi=chi,
        Ku=Ku, #Introduce changes here
        alpha=alpha,
        dbpar_energy_weight=1.0 / alpha,
    )


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def characteristic_speeds(params: Any) -> list[float]:
    """Return parallel linear speeds relevant to the CFL estimate."""

    p = derived_parameters(params)
    return [p.vA, p.vA * np.sqrt(p.alpha)]


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space ideal RHS of the homogeneous four-field system."""

    p = derived_parameters(params)

    psi_hat = state["psi"]
    omega_hat = state["omega"]
    upar_hat = state["upar"]
    dbpar_hat = state["dbpar"]

    phi_hat = derive_phi_hat(omega_hat, grid)
    lap_psi_hat = lap_perp(psi_hat, grid)

    rhs_state = state.zeros_like() if out is None else out
    rhs_state.fill_zero()

    rhs_psi = rhs_state["psi"]
    rhs_psi[...] = p.vA * dz(phi_hat, grid)
    rhs_psi[...] -= poisson_bracket(
        phi_hat,
        psi_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )

    rhs_omega = rhs_state["omega"]
    rhs_omega[...] = p.vA * dz(lap_psi_hat, grid)
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
    rhs_dbpar[...] = p.alpha * dz(upar_hat, grid)
    rhs_dbpar[...] -= poisson_bracket(
        phi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] += p.alpha * poisson_bracket(
        psi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_dbpar[...] -= p.Ku * p.alpha * dy(psi_hat, grid)

    rhs_upar = rhs_state["upar"]
    rhs_upar[...] = (p.vA**2) * dz(dbpar_hat, grid)
    rhs_upar[...] -= poisson_bracket(
        phi_hat,
        upar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_upar[...] += (p.vA**2) * poisson_bracket(
        psi_hat,
        dbpar_hat,
        grid,
        fft,
        workspace,
        mask=dealias_mask,
    )
    rhs_upar[...] += p.Ku * dy(phi_hat, grid)

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 4x4 linear matrix for one Fourier mode.

    The field order is `[psi, omega, upar, dbpar]`. For `k_perp = 0`, the
    `psi/omega` Alfvénic block is set to zero because the inverse perpendicular
    Laplacian is not meaningful there in the RMHD subspace.
    """

    p = derived_parameters(params)
    matrix = np.zeros((4, 4), dtype=np.complex128)
    ikz = 1j * float(kz)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -p.vA * ikz / kperp2
        matrix[1, 0] = -p.vA * ikz * kperp2
        matrix[2,1] = -p.Ku * 1j * float(ky) / kperp2
     
    matrix[2, 3] = (p.vA**2) * ikz
    matrix[3, 2] = p.alpha * ikz
    matrix[3, 0] = -p.Ku * p.alpha * 1j * float(ky)
    return matrix


def _dissipation_spec_for_field(
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None,
) -> Mapping[str, float | int]:
    if dissipation_spec is not None:
        return dissipation_spec[field_name]
    if isinstance(params, Mapping):
        return params["dissipation"][field_name]
    return getattr(params, "dissipation")[field_name]


def dissipation_operator(
    grid: Any,
    params: Any,
    field_name: str,
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> Any:
    """Return the nonnegative diagonal damping operator `D_i(k)` for one field."""

    spec = _dissipation_spec_for_field(params, field_name, dissipation_spec)
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
    dissipation_spec: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Build the diagonal damping operators for all evolved fields."""

    names = FIELD_NAMES if field_names is None else field_names
    return {
        name: dissipation_operator(grid, params, name, dissipation_spec=dissipation_spec)
        for name in names
    }


def perpendicular_energy_spectra(
    state: State,
    grid: Any,
    backend: Any,
    *,
    bin_width: float | None = None,
    params: Any | None = None,
) -> dict[str, np.ndarray]:
    """Return the standard S09 perpendicular shell spectra."""

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    kperp2 = grid.kperp2

    kperp, u_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(phi_hat) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, b_perp = perpendicular_shell_spectrum(
        0.5 * kperp2 * (xp.abs(state["psi"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, upar = perpendicular_shell_spectrum(
        0.5 * (xp.abs(state["upar"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, dbpar = perpendicular_shell_spectrum(
        0.5 * p.dbpar_energy_weight * (xp.abs(state["dbpar"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    # Elsasser fields z ± = u_perp ∓ b_perp/sqrt(4 pi rho0) = z_hat x grad_perp(phi ∓ psi).
    # The 1/4 weight is the standard pseudo-energy normalization, so that
    # z_plus + z_minus = u_perp + b_perp shell by shell.
    _, z_plus = perpendicular_shell_spectrum(
        0.25 * kperp2 * (xp.abs(phi_hat - state["psi"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    _, z_minus = perpendicular_shell_spectrum(
        0.25 * kperp2 * (xp.abs(phi_hat + state["psi"]) ** 2),
        grid,
        backend,
        bin_width=bin_width,
    )
    return {
        "kperp": kperp,
        "u_perp": u_perp,
        "b_perp": b_perp,
        "du_par": upar,
        "db_par": dbpar,
        "z_plus": z_plus,
        "z_minus": z_minus,
    }


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the S09 total energy.

    The Alfvénic fields are measured as physical perpendicular amplitudes:
    `u_perp ~ grad_perp phi` and `b_perp ~ grad_perp psi`. Therefore the modal
    density uses `k_perp^2 |phi_hat|^2` and `k_perp^2 |psi_hat|^2`, not raw
    potential amplitudes.

    In code variables:

    `E = 0.5 * (|grad_perp phi|^2 + |grad_perp psi|^2 + |upar|^2`
    `           + alpha^(-1) |dbpar|^2`.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    return (
        0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
        + 0.5 * xp.abs(state["upar"]) ** 2
        + 0.5 * p.dbpar_energy_weight * xp.abs(state["dbpar"]) ** 2
    )

def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the volume-averaged total energy for this equation set."""

    density_hat = total_energy_modal_density(state, grid, backend, params)
    return modal_average(density_hat, grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any) -> float:
    """Return the S09 Alfvenic energy partition."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)
    return modal_average(density_hat, grid, backend)


def _unweighted_field_energy(field_hat: Any, grid: Any, backend: Any) -> float:
    return modal_average(0.5 * backend.xp.abs(field_hat) ** 2, grid, backend)

def total_energy_shear_rhs(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the signed ideal shear contribution `(put in the equation here later)`."""

    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    v_hat = dy(phi_hat, grid) * p.Ku
    u_hat = -dy(state["psi"], grid) * p.Ku * (p.vA**2)
    return modal_inner_product_average(v_hat, state["upar"], grid, backend) + modal_inner_product_average(u_hat, state["dbpar"], grid, backend)

def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`.

    Sign convention:

    `d_t E = forcing + other_terms + dissipation`

    so this term is negative when diagonal damping removes energy. The weights
    match :func:`total_energy_modal_density` exactly.
    """

    xp = backend.xp
    p = derived_parameters(params)
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
        - linear_ops["upar"] * xp.abs(state["upar"]) ** 2
        - p.dbpar_energy_weight * linear_ops["dbpar"] * xp.abs(state["dbpar"]) ** 2
    )
    return modal_average(density_hat, grid, backend)


def compute_conserved_quantity_budgets(
    state: State,
    *,
    grid: Any,
    backend: Any,
    params: Any,
    linear_ops: dict[str, Any] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return conserved-quantity values plus named signed RHS contributions. The total energy is not conserved ideally; the explicit source is
    stored as `shear` with sign convention

    `d_t E = shear + dissipation + forcing + ...`."""
#Not sure whether to make changes here, compare with stratification
    rhs_terms: dict[str, float] = {
        "shear": total_energy_shear_rhs(state, grid, backend, params),
    }
    if linear_ops is not None:
        rhs_terms["dissipation"] = total_energy_dissipation_rhs(
            state,
            grid,
            backend,
            linear_ops,
            params,
        )
    if extra_rhs_terms is not None:
        rhs_terms.update(
            {
                name: float(value)
                for name, value in extra_rhs_terms.get("total_energy", {}).items()
            }
        )

    return {
        "total_energy": {
            "value": total_energy(state, grid, backend, params),
            "rhs_terms": rhs_terms,
        }
    }


def compute_equation_scalar_diagnostics(
    state: State,
    *,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any | None = None,
    linear_ops: dict[str, Any] | None = None,
    budget_rhs_terms: dict[str, dict[str, float]] | None = None,
    extra_rhs_terms: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Return S09-specific scalar diagnostics.

    Equation modules own scientifically meaningful scalar diagnostics. For a
    new equation set, provide this function and include the standard
    `total_energy` / `total_energy_rhs_*` names so generic budget plotting
    tools can operate without knowing equation-specific details.
    """

    alfvenic = alfvenic_energy(state, grid, backend)
    upar = _unweighted_field_energy(state["upar"], grid, backend)
    dbpar = _unweighted_field_energy(state["dbpar"], grid, backend)
    diagnostics = {
        "alfvenic_energy": alfvenic,
        "upar_energy": upar,
        "dbpar_energy": dbpar,
        "total_energy_proxy": alfvenic + upar + dbpar,
    }

    budgets = compute_conserved_quantity_budgets(
        state,
        grid=grid,
        backend=backend,
        params=params,
        linear_ops=linear_ops,
        extra_rhs_terms=extra_rhs_terms,
    )
    rhs_terms = budgets["total_energy"].setdefault("rhs_terms", {})
    if budget_rhs_terms is not None and "total_energy" in budget_rhs_terms:
        rhs_terms.clear()
        rhs_terms.update({name: float(value) for name, value in budget_rhs_terms["total_energy"].items()})
    rhs_terms.setdefault("dissipation", 0.0)
    rhs_terms.setdefault("forcing", 0.0)
    diagnostics.update(flatten_conserved_quantity_budgets(budgets))
    return diagnostics
