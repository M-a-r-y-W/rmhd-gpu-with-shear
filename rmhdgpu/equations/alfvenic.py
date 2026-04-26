"""Two-field Alfvénic RMHD equation set.

This is the Alfvénic subsector of the homogeneous S09 system with only the
fields `[psi, omega]`, where `phi = inv_lap_perp(omega)`.

The ideal equations are

- `psi_t = vA * dz(phi) - {phi, psi}`
- `omega_t = vA * dz(lap_perp psi) - {phi, omega} + {psi, lap_perp psi}`

with `omega = lap_perp(phi)` and `j = -lap_perp(psi)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.diagnostics.budget import flatten_conserved_quantity_budgets
from rmhdgpu.diagnostics.scalar import STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO
from rmhdgpu.diagnostics.spectra import perpendicular_shell_spectrum
from rmhdgpu.fourier_diagnostics import modal_average
from rmhdgpu.operators import dz, inv_lap_perp, lap_perp, poisson_bracket
from rmhdgpu.state import State


EQUATION_SET_NAME = "alfvenic"
FIELD_NAMES = ["psi", "omega"]
DEFAULT_INITIAL_CONDITION = "alfven_mode"

SCALAR_DIAGNOSTIC_INFO = {
    **STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO,
    "alfvenic_energy": "Total two-field Alfvenic energy: 0.5 <|grad phi|^2 + |grad psi|^2>.",
}


@dataclass(frozen=True, slots=True)
class AlfvenicParameters:
    """Scalar parameters used by the two-field Alfvénic system."""

    vA: float


def _param_float(params: Any, name: str) -> float:
    if isinstance(params, Mapping):
        return float(params[name])
    return float(getattr(params, name))


def derived_parameters(params: Any) -> AlfvenicParameters:
    """Return the compact scalar parameter block for this equation set."""

    return AlfvenicParameters(vA=_param_float(params, "vA"))


def derive_phi_hat(omega_hat: Any, grid: Any) -> Any:
    """Return `phi_hat = inv_lap_perp(omega_hat)`."""

    return inv_lap_perp(omega_hat, grid)


def derive_j_hat(psi_hat: Any, grid: Any) -> Any:
    """Return `j_hat = -lap_perp(psi_hat) = +k_perp^2 psi_hat`."""

    return -lap_perp(psi_hat, grid)


def characteristic_speeds(params: Any) -> list[float]:
    """Return parallel linear speeds relevant to the CFL estimate."""

    return [derived_parameters(params).vA]


def ideal_rhs(
    state: State,
    grid: Any,
    fft: Any,
    workspace: Any,
    params: Any,
    dealias_mask: Any | None = None,
    out: State | None = None,
) -> State:
    """Return the Fourier-space ideal RHS of the two-field Alfvénic system."""

    p = derived_parameters(params)
    psi_hat = state["psi"]
    omega_hat = state["omega"]
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

    return rhs_state


def linear_matrix(kx: float, ky: float, kz: float, params: Any) -> np.ndarray:
    """Return the 2x2 linear matrix for one Fourier mode.

    Field order is `[psi, omega]`. For `k_perp = 0`, the matrix is set to zero
    because `phi = inv_lap_perp(omega)` is not meaningful on that subspace.
    """

    p = derived_parameters(params)
    matrix = np.zeros((2, 2), dtype=np.complex128)
    ikz = 1j * float(kz)
    kperp2 = float(kx) ** 2 + float(ky) ** 2

    if kperp2 > 0.0:
        matrix[0, 1] = -p.vA * ikz / kperp2
        matrix[1, 0] = -p.vA * ikz * kperp2
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
    """Return perpendicular shell spectra for the Alfvénic energy pieces."""

    xp = backend.xp
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
    return {"kperp": kperp, "u_perp": u_perp, "b_perp": b_perp}


def total_energy_modal_density(state: State, grid: Any, backend: Any, params: Any) -> Any:
    """Return the modal quadratic density for the two-field Alfvénic energy."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    return 0.5 * grid.kperp2 * (xp.abs(phi_hat) ** 2 + xp.abs(state["psi"]) ** 2)


def total_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return `0.5 <|grad phi|^2 + |grad psi|^2>`."""

    return modal_average(total_energy_modal_density(state, grid, backend, params), grid, backend)


def alfvenic_energy(state: State, grid: Any, backend: Any, params: Any) -> float:
    """Return the total two-field Alfvénic energy."""

    return total_energy(state, grid, backend, params)


def total_energy_dissipation_rhs(
    state: State,
    grid: Any,
    backend: Any,
    linear_ops: dict[str, Any],
    params: Any,
) -> float:
    """Return the signed dissipative contribution to `d_t E`."""

    xp = backend.xp
    phi_hat = derive_phi_hat(state["omega"], grid)
    density_hat = (
        -linear_ops["omega"] * grid.kperp2 * (xp.abs(phi_hat) ** 2)
        - linear_ops["psi"] * grid.kperp2 * (xp.abs(state["psi"]) ** 2)
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
    """Return total energy plus named signed RHS contributions."""

    rhs_terms: dict[str, float] = {}
    if linear_ops is not None:
        rhs_terms["dissipation"] = total_energy_dissipation_rhs(state, grid, backend, linear_ops, params)
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
    """Return Alfvénic equation-set scalar diagnostics."""

    diagnostics = {
        "alfvenic_energy": alfvenic_energy(state, grid, backend, params),
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
