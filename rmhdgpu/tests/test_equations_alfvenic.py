from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import alfvenic
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step
from rmhdgpu.workspace import Workspace


def _build_context() -> tuple[Config, object, object, FFTManager, Workspace, object]:
    config = Config(equation_set="alfvenic", Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.3)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, workspace, mask


def _advance(
    state: State,
    *,
    steps: int,
    dt: float,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> State:
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    current = state
    for _ in range(steps):
        current = ssprk3_step(current, dt, alfvenic.ideal_rhs, rhs_kwargs=rhs_kwargs)
    return current


def _scaled_state(state: State, factor: complex) -> State:
    out = state.copy()
    for name in out.field_names:
        out[name][...] *= factor
    return out


def test_alfvenic_rhs_zero_state() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state = State(grid, backend, field_names=alfvenic.FIELD_NAMES)

    rhs_state = alfvenic.ideal_rhs(state, grid, fft, workspace, config, dealias_mask=mask)

    for name in rhs_state.field_names:
        np.testing.assert_allclose(backend.to_numpy(rhs_state[name]), 0.0, atol=0.0, rtol=0.0)


def test_alfvenic_linear_matrix_eigenvalues_match_dispersion() -> None:
    config, _, _, _, _, _ = _build_context()
    kz = 1.25
    matrix = alfvenic.linear_matrix(kx=1.0, ky=2.0, kz=kz, params=config)
    eigenvalues = np.sort_complex(np.linalg.eigvals(matrix))
    expected = np.sort_complex(np.array([1j * config.vA * kz, -1j * config.vA * kz], dtype=np.complex128))
    np.testing.assert_allclose(eigenvalues, expected, atol=1.0e-12, rtol=1.0e-12)


def test_alfvenic_single_mode_matches_exact_linear_evolution() -> None:
    config, backend, grid, fft, workspace, mask = _build_context()
    state0 = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": [1, 2, 1], "amplitude": 0.3, "branch": "plus"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    dt = 5.0e-3
    steps = 20
    total_time = steps * dt
    kz = backend.scalar_to_float(grid.kz[0, 0, 1])
    lambda_mode = 1j * config.vA * kz

    evolved = _advance(state0, steps=steps, dt=dt, config=config, grid=grid, fft=fft, workspace=workspace, mask=mask)
    exact = _scaled_state(state0, np.exp(lambda_mode * total_time))

    for name in state0.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(evolved[name]),
            backend.to_numpy(exact[name]),
            atol=2.0e-9,
            rtol=2.0e-9,
            err_msg=f"Mismatch in alfvenic exact evolution for field {name}.",
        )


def test_alfvenic_alfven_mode_energy_matches_amplitude_squared() -> None:
    config, backend, grid, fft, _, mask = _build_context()
    amplitude = 0.2
    state = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": [1, 1, 1], "amplitude": amplitude, "branch": "plus"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    np.testing.assert_allclose(
        alfvenic.total_energy(state, grid, backend, config),
        amplitude**2,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
