from __future__ import annotations

import numpy as np
import pytest

from rmhdgpu import NonFiniteStateError, check_state_finite
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until


def _build_context() -> tuple[Config, object, object, FFTManager]:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        dt_init=0.1,
        runtime_check_every=2,
        fail_on_nonfinite=True,
    )
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    return config, backend, grid, fft


def _finite_state(grid: object, backend: object) -> State:
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    state["psi"][1, 1, 1] = 1.0 + 0.5j
    state["omega"][1, 1, 1] = -2.0 + 0.25j
    return state


def _zero_linear_ops(state: State) -> dict[str, object]:
    return {
        name: state.backend.zeros(state.grid.fourier_shape, dtype=state.grid.real_dtype)
        for name in state.field_names
    }


def _zero_rhs(state: State, **kwargs) -> State:
    return state.zeros_like()


def test_check_state_finite_passes_for_finite_state() -> None:
    _, backend, grid, _ = _build_context()
    state = _finite_state(grid, backend)

    check_state_finite(state, backend, time=0.0, step=0, context="unit test")


def test_check_state_finite_raises_on_nan() -> None:
    _, backend, grid, _ = _build_context()
    state = _finite_state(grid, backend)
    state["psi"][0, 0, 0] = np.nan + 0.0j

    with pytest.raises(NonFiniteStateError) as excinfo:
        check_state_finite(state, backend, time=1.25, step=7, context="time integration")

    message = str(excinfo.value)
    assert "psi" in message
    assert "step 7" in message
    assert "t=1.25" in message
    assert "numerical instability" in message


def test_check_state_finite_raises_on_inf() -> None:
    _, backend, grid, _ = _build_context()
    state = _finite_state(grid, backend)
    state["omega"][0, 0, 0] = np.inf + 0.0j

    with pytest.raises(NonFiniteStateError) as excinfo:
        check_state_finite(state, backend, step=3, context="post-step check")

    message = str(excinfo.value)
    assert "omega" in message
    assert "step 3" in message
    assert "post-step check" in message


def test_evolution_loop_stops_on_nonfinite() -> None:
    config, backend, grid, fft = _build_context()
    state = _finite_state(grid, backend)
    linear_ops = _zero_linear_ops(state)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": None,
        "params": config,
        "dealias_mask": None,
    }
    call_count = {"n": 0}

    def bad_stepper(current: State, dt: float, ideal_rhs_func: object, linear_ops: object, rhs_kwargs=None) -> State:
        call_count["n"] += 1
        out = current.copy()
        if call_count["n"] >= 3:
            out["psi"][0, 0, 0] = np.nan + 0.0j
        return out

    with pytest.raises(NonFiniteStateError) as excinfo:
        evolve_until(
            state,
            0.4,
            _zero_rhs,
            linear_ops,
            rhs_kwargs=rhs_kwargs,
            params=config,
            fixed_dt=0.1,
            check_every=1,
            stepper_func=bad_stepper,
        )

    message = str(excinfo.value)
    assert "psi" in message
    assert "step 3" in message
    assert "time integration" in message


def test_runtime_check_interval_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    config, backend, grid, fft = _build_context()
    config.runtime_check_every = 2
    config.progress_output_every = None
    state = _finite_state(grid, backend)
    linear_ops = _zero_linear_ops(state)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": None,
        "params": config,
        "dealias_mask": None,
    }
    calls: list[tuple[int | None, float | None, str]] = []

    def spy_check(current: State, backend_obj: object, *, time=None, step=None, context="") -> None:
        calls.append((step, time, context))

    def identity_stepper(current: State, dt: float, ideal_rhs_func: object, linear_ops: object, rhs_kwargs=None) -> State:
        return current.copy()

    monkeypatch.setattr("rmhdgpu.steppers.check_state_finite", spy_check)

    final_state, info = evolve_until(
        state,
        0.3,
        _zero_rhs,
        linear_ops,
        rhs_kwargs=rhs_kwargs,
        params=config,
        fixed_dt=0.1,
        stepper_func=identity_stepper,
    )

    assert final_state.field_names == state.field_names
    assert info["steps"] == 3
    assert calls == [
        (0, 0.0, "time integration startup"),
        (2, pytest.approx(0.2), "time integration"),
        (3, pytest.approx(0.3), "time integration"),
    ]


def test_progress_output_interval_respected(capsys: pytest.CaptureFixture[str]) -> None:
    config, backend, grid, fft = _build_context()
    config.progress_output_every = 2
    state = _finite_state(grid, backend)
    linear_ops = _zero_linear_ops(state)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": None,
        "params": config,
        "dealias_mask": None,
    }

    def identity_stepper(current: State, dt: float, ideal_rhs_func: object, linear_ops: object, rhs_kwargs=None) -> State:
        return current.copy()

    final_state, info = evolve_until(
        state,
        0.3,
        _zero_rhs,
        linear_ops,
        rhs_kwargs=rhs_kwargs,
        params=config,
        fixed_dt=0.1,
        stepper_func=identity_stepper,
    )

    captured = capsys.readouterr()
    assert final_state.field_names == state.field_names
    assert info["steps"] == 3
    assert "progress step=2" in captured.out
    assert "progress step=3" in captured.out
    assert "progress step=1" not in captured.out
