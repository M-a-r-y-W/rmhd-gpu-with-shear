"""Linear eigenmode constructors for the low-beta stratified equation set.

These helpers are equation-specific and intentionally live outside
`builtin.py`. The registry in `builtin.py` should remain a thin user-facing
dispatch layer, while reusable eigenmode construction for each equation set
belongs in a small `eigenmodes_<equation>.py` module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rmhdgpu.equations import get_equation_module
from rmhdgpu.state import State


def _select_mode_index(eigenvalues: np.ndarray, mode: str) -> int:
    if mode == "unstable_growing":
        return int(np.argmax(eigenvalues.real))
    if mode == "unstable_decaying":
        return int(np.argmin(eigenvalues.real))
    if mode == "stable_plus":
        return int(np.argmax(eigenvalues.imag))
    if mode == "stable_minus":
        return int(np.argmin(eigenvalues.imag))
    raise ValueError(f"Unknown low-beta eigenmode branch {mode!r}.")


def low_beta_stratified_mode_state(
    grid: Any,
    backend: Any,
    field_names: Sequence[str] | None,
    k_indices: Sequence[int],
    amplitude: complex | float = 1.0,
    mode: str = "unstable_growing",
    params: Any | None = None,
) -> State:
    """Return a single low-beta stratified linear eigenmode.

    The eigenvector is selected numerically from the equation module's
    `linear_matrix(...)`. This keeps the initializer tied to the same linear
    representation used by tests and avoids hand-coded branch relations.
    """

    if params is None:
        raise ValueError("low_beta_stratified_mode_state requires params so vA and N2 are defined.")
    if len(k_indices) != 3:
        raise ValueError(f"k_indices must have length 3; got {k_indices!r}.")

    equation_module = get_equation_module("low_beta_stratified")
    names = list(equation_module.FIELD_NAMES if field_names is None else field_names)
    state = State(grid, backend, field_names=names)

    ix_raw, iy_raw, iz = (int(k_indices[0]), int(k_indices[1]), int(k_indices[2]))
    if iz < 0 or iz > grid.Nz // 2:
        raise ValueError(f"k_indices[2] must satisfy 0 <= kz <= Nz//2; got {iz}.")
    ix = ix_raw % grid.Nx
    iy = iy_raw % grid.Ny

    kx = backend.scalar_to_float(grid.kx[ix, 0, 0])
    ky = backend.scalar_to_float(grid.ky[0, iy, 0])
    kz = backend.scalar_to_float(grid.kz[0, 0, iz])

    matrix = equation_module.linear_matrix(kx=kx, ky=ky, kz=kz, params=params)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    selected = _select_mode_index(eigenvalues, mode)

    vector = eigenvectors[:, selected]
    scale = np.max(np.abs(vector))
    if scale <= 0.0:
        raise ValueError("Selected low-beta eigenvector has zero amplitude.")
    vector = amplitude * vector / scale

    for component, field_name in enumerate(equation_module.FIELD_NAMES):
        state[field_name][ix, iy, iz] = vector[component]
    return state
