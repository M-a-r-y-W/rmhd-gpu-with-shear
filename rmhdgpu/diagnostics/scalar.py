"""Simple scalar diagnostics available before a full solver exists."""

from __future__ import annotations

from typing import Any

from rmhdgpu.diagnostics.alfvenic import alfvenic_energy


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


def compute_energy_diagnostics(
    state: Any,
    grid: Any,
    fft: Any,
    backend: Any,
    params: Any,
    workspace: Any | None = None,
    equation_module: Any | None = None,
) -> dict[str, float]:
    """Return generic and equation-specific quadratic energy diagnostics."""

    xp = backend.xp
    diagnostics: dict[str, float] = {}

    if "psi" in state.field_names and "omega" in state.field_names:
        diagnostics["alfvenic_energy"] = alfvenic_energy(state, grid, fft)

    scratch_names = ["r0", "r1", "r2", "r3", "r4"]
    proxy_total = diagnostics.get("alfvenic_energy", 0.0)
    for index, field_name in enumerate(state.field_names):
        if field_name in {"psi", "omega"}:
            continue
        if workspace is None:
            field_real = fft.c2r(state[field_name])
        else:
            field_real = fft.c2r(state[field_name], out=workspace.real[scratch_names[index % len(scratch_names)]])
        value = backend.scalar_to_float(0.5 * xp.mean(field_real**2))
        if field_name == "s":
            diagnostics["entropy_variance"] = value
        else:
            diagnostics[f"{field_name}_energy"] = value
        proxy_total += value

    if diagnostics:
        diagnostics["total_energy_proxy"] = proxy_total
    if equation_module is not None:
        diagnostics["total_energy"] = equation_module.total_energy(state, grid, backend, params)
    elif hasattr(params, "equation_set"):
        from rmhdgpu.equations import get_equation_module

        module = get_equation_module(getattr(params, "equation_set"))
        diagnostics["total_energy"] = module.total_energy(state, grid, backend, params)
    return diagnostics
