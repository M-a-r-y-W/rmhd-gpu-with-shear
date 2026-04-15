from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state, get_initial_condition_builder, list_initial_condition_types
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.operators import lap_perp


def _build_context() -> tuple[Config, object, object, FFTManager, object]:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend)
    return config, backend, grid, fft, mask


def test_known_initial_conditions_are_registered() -> None:
    assert {"zero", "alfven_mode", "aw_packet", "decaying_low_modes"} <= set(list_initial_condition_types())


def test_unknown_initial_condition_gives_helpful_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_initial_condition_builder("does_not_exist")

    message = str(excinfo.value)
    assert "Unknown initial condition type 'does_not_exist'" in message
    for name in ("zero", "alfven_mode", "aw_packet", "decaying_low_modes"):
        assert name in message


def test_zero_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "zero",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    for name in state.field_names:
        np.testing.assert_allclose(backend.to_numpy(state[name]), 0.0)


def test_alfven_mode_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "alfven_mode",
        parameters={"k_indices": [1, 1, 1], "amplitude": 0.2, "branch": "plus"},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    psi_hat = backend.to_numpy(state["psi"])
    omega_hat = backend.to_numpy(state["omega"])

    assert np.isfinite(psi_hat).all()
    assert np.count_nonzero(np.abs(psi_hat) > 0.0) == 1
    np.testing.assert_allclose(omega_hat, backend.to_numpy(lap_perp(state["psi"], grid)))


def test_aw_packet_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "aw_packet",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    psi_hat = backend.to_numpy(state["psi"])
    psi_real = backend.to_numpy(fft.c2r(state["psi"]))

    assert np.isfinite(psi_hat).all()
    assert np.isfinite(psi_real).all()
    assert np.max(np.abs(psi_real)) > 0.0
    np.testing.assert_allclose(backend.to_numpy(state["omega"]), backend.to_numpy(lap_perp(state["psi"], grid)))


def test_decaying_low_modes_initial_condition() -> None:
    config, backend, grid, fft, mask = _build_context()
    state = build_initial_state(
        "decaying_low_modes",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    for name in config.field_names:
        field = backend.to_numpy(state[name])
        assert np.isfinite(field).all()
        assert np.max(np.abs(field)) > 0.0


def test_decaying_low_modes_initial_condition_parameter_overrides() -> None:
    config, backend, grid, fft, mask = _build_context()
    default_state = build_initial_state(
        "decaying_low_modes",
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    override_state = build_initial_state(
        "decaying_low_modes",
        parameters={"psi_seed": 21, "psi_amplitude": 0.12},
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )

    assert np.isfinite(backend.to_numpy(override_state["psi"])).all()
    assert not np.allclose(
        backend.to_numpy(default_state["psi"]),
        backend.to_numpy(override_state["psi"]),
    )
