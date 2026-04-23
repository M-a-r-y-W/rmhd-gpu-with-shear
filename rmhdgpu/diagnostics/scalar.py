"""Scalar diagnostics helpers.

This module owns only generic, equation-independent scalar diagnostics. These
are simple per-field real-space statistics that make sense for any evolved
field list:

- `<field>_mean`
- `<field>_rms`
- `<field>_max_abs`

Scientifically meaningful quantities such as total energy, energy partitions,
and budget/source terms belong in the active equation module through
`compute_equation_scalar_diagnostics(...)`. Keeping that split avoids encoding
S09-specific assumptions in generic diagnostics code.
"""

from __future__ import annotations

from typing import Any


GENERIC_FIELD_SCALAR_DIAGNOSTIC_INFO = {
    "<field>_mean": "Volume mean of one evolved real-space field.",
    "<field>_rms": "Root-mean-square amplitude of one evolved real-space field.",
    "<field>_max_abs": "Maximum absolute real-space amplitude of one evolved field.",
}

STANDARD_ENERGY_SCALAR_DIAGNOSTIC_INFO = {
    "total_energy": "Equation-module total energy used by budget plotting tools.",
    "total_energy_rhs_total": "Sum of saved signed RHS contributions to d_t total_energy.",
    "total_energy_rhs_dissipation": "Signed dissipative contribution to d_t total_energy; negative when damping removes energy.",
    "total_energy_rhs_forcing": "Signed forcing contribution to d_t total_energy; based on the run's forcing-kick accounting.",
    "total_energy_rhs_<name>": "Pattern for any additional signed total-energy RHS contribution supplied by an equation set.",
}


def compute_scalar_diagnostics(
    state: Any,
    grid: Any,
    fft: Any,
    backend: Any,
    workspace: Any | None = None,
) -> dict[str, float]:
    """Compute basic real-space scalar diagnostics for each field.

    Each Fourier field is inverse transformed to real space, then the following
    quantities are reported:

    - mean
    - RMS
    - maximum absolute value
    """

    xp = backend.xp
    diagnostics: dict[str, float] = {}
    scratch_real = None if workspace is None else workspace.real.get("r0")

    for name in state.field_names:
        field_real = fft.c2r(state[name], out=scratch_real)
        diagnostics[f"{name}_mean"] = backend.scalar_to_float(xp.mean(field_real))
        diagnostics[f"{name}_rms"] = backend.scalar_to_float(xp.sqrt(xp.mean(field_real**2)))
        diagnostics[f"{name}_max_abs"] = backend.scalar_to_float(xp.max(xp.abs(field_real)))

    return diagnostics


compute_field_scalar_diagnostics = compute_scalar_diagnostics


def compute_energy_diagnostics(
    state: Any,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any | None = None,
    equation_module: Any | None = None,
) -> dict[str, float]:
    """Compatibility wrapper for equation-specific scalar diagnostics.

    New code should prefer calling the selected equation module's
    `compute_equation_scalar_diagnostics(...)` directly. This wrapper remains
    for examples/profiling/tests that historically imported
    `compute_energy_diagnostics`.
    """

    module = equation_module
    if module is None and hasattr(params, "equation_set"):
        from rmhdgpu.equations import get_equation_module

        module = get_equation_module(getattr(params, "equation_set"))
    if module is None:
        return {}
    return module.compute_equation_scalar_diagnostics(
        state,
        grid=grid,
        fft=fft,
        backend=backend,
        params=params,
        workspace=workspace,
    )
