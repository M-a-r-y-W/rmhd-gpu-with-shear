from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.initconds.testing import single_mode_field
from rmhdgpu.initconds.testing import random_spectrum_test_parameters
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import if_ssprk3_step, ssprk3_step
from rmhdgpu.workspace import Workspace


def _zero_rhs(state: State, **kwargs) -> State:
    return state.zeros_like()


def _build_context(config: Config) -> tuple[object, object, FFTManager, Workspace, object]:
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return backend, grid, fft, workspace, mask


@pytest.mark.parametrize(
    "field_name,dissipation,mode_indices,total_time",
    [
        ("psi", {"nu_perp": 5.0e-3, "nu_par": 0.0, "n_perp": 2, "n_par": 1}, (1, 2, 1), 0.3),
        ("upar", {"nu_perp": 0.0, "nu_par": 4.0e-2, "n_perp": 2, "n_par": 1}, (1, 1, 2), 0.25),
        ("dbpar", {"nu_perp": 3.0e-3, "nu_par": 2.0e-2, "n_perp": 2, "n_par": 1}, (2, 1, 2), 0.2),
    ],
)
def test_single_mode_decay_matches_exact_exponential(
    field_name: str,
    dissipation: dict[str, float | int],
    mode_indices: tuple[int, int, int],
    total_time: float,
) -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    config.dissipation[field_name].update(dissipation)
    backend, grid, _, _, _ = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)

    state = State(grid, backend, field_names=config.field_names)
    state[field_name][...] = single_mode_field(grid, backend, mode_indices, amplitude=1.0)
    amplitude0 = complex(backend.to_numpy(state[field_name])[mode_indices])
    damping_rate = float(backend.to_numpy(linear_ops[field_name])[mode_indices])

    dt = 0.05
    steps = int(round(total_time / dt))
    current = state
    for _ in range(steps):
        current = if_ssprk3_step(current, dt, _zero_rhs, linear_ops)

    amplitude_final = complex(backend.to_numpy(current[field_name])[mode_indices])
    exact = amplitude0 * np.exp(-damping_rate * total_time)

    np.testing.assert_allclose(amplitude_final, exact, atol=1.0e-13, rtol=1.0e-13)


def test_anisotropic_operator_distinguishes_kperp_and_kz() -> None:
    config = Config(Nx=12, Ny=12, Nz=12, backend="numpy")
    config.dissipation["psi"].update({"nu_perp": 0.7, "nu_par": 0.4, "n_perp": 2, "n_par": 1})
    backend, grid, _, _, _ = _build_context(config)
    operator = backend.to_numpy(s09.dissipation_operator(grid, config, "psi"))

    same_kperp_low_kz = (1, 1, 1)
    same_kperp_high_kz = (1, 1, 2)
    low_kperp_same_kz = (1, 1, 1)
    high_kperp_same_kz = (2, 1, 1)

    rate_same_kperp_low = operator[same_kperp_low_kz]
    rate_same_kperp_high = operator[same_kperp_high_kz]
    rate_low_kperp = operator[low_kperp_same_kz]
    rate_high_kperp = operator[high_kperp_same_kz]

    assert rate_same_kperp_high > rate_same_kperp_low
    assert rate_high_kperp > rate_low_kperp

    expected_low = (
        0.7 * backend.to_numpy(grid.kperp2)[same_kperp_low_kz] ** 2
        + 0.4 * backend.to_numpy(grid.kpar2)[same_kperp_low_kz]
    )
    expected_high = (
        0.7 * backend.to_numpy(grid.kperp2)[high_kperp_same_kz] ** 2
        + 0.4 * backend.to_numpy(grid.kpar2)[high_kperp_same_kz]
    )

    np.testing.assert_allclose(rate_same_kperp_low, expected_low, atol=1.0e-14, rtol=1.0e-14)
    np.testing.assert_allclose(rate_high_kperp, expected_high, atol=1.0e-14, rtol=1.0e-14)


def test_zero_dissipation_reduces_to_ideal() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    backend, grid, fft, workspace, mask = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = build_initial_state(
        "random_spectrum",
        parameters=random_spectrum_test_parameters(0.1),
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    ideal_step = ssprk3_step(state, 0.02, s09.ideal_rhs, rhs_kwargs=rhs_kwargs)
    if_step = if_ssprk3_step(state, 0.02, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)

    for name in state.field_names:
        np.testing.assert_allclose(
            backend.to_numpy(if_step[name]),
            backend.to_numpy(ideal_step[name]),
            atol=1.0e-13,
            rtol=1.0e-13,
            err_msg=f"Integrating-factor step should match ideal RK3 when dissipation is zero for {name}.",
        )
