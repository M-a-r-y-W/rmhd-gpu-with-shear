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
    workspace: Any | None = None,
) -> dict[str, float]:
    """Return a small set of quadratic energy-like diagnostics.

    The returned values are volume averages in real space:

    - `alfvenic_energy = 0.5 <|grad_perp phi|^2 + |grad_perp psi|^2>`
    - `upar_energy = 0.5 <upar^2>`
    - `dbpar_energy = 0.5 <dbpar^2>`
    - `entropy_variance = 0.5 <s^2>`
    - `total_energy_proxy`, the sum of the above pieces
    """

    xp = backend.xp
    diagnostics = {
        "alfvenic_energy": alfvenic_energy(state, grid, fft),
    }

    if workspace is None:
        upar_real = fft.c2r(state["upar"])
        dbpar_real = fft.c2r(state["dbpar"])
        entropy_real = fft.c2r(state["s"])
    else:
        upar_real = fft.c2r(state["upar"], out=workspace.real["r0"])
        dbpar_real = fft.c2r(state["dbpar"], out=workspace.real["r1"])
        entropy_real = fft.c2r(state["s"], out=workspace.real["r2"])

    diagnostics["upar_energy"] = backend.scalar_to_float(0.5 * xp.mean(upar_real**2))
    diagnostics["dbpar_energy"] = backend.scalar_to_float(0.5 * xp.mean(dbpar_real**2))
    diagnostics["entropy_variance"] = backend.scalar_to_float(0.5 * xp.mean(entropy_real**2))
    diagnostics["total_energy_proxy"] = (
        diagnostics["alfvenic_energy"]
        + diagnostics["upar_energy"]
        + diagnostics["dbpar_energy"]
        + diagnostics["entropy_variance"]
    )
    return diagnostics
