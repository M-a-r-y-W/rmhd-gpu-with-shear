from __future__ import annotations

import pytest

from rmhdgpu import NonFiniteStateError, check_state_finite
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until


cupy = pytest.importorskip("cupy")
try:
    cupy.zeros((1,), dtype=cupy.float64)
except Exception as exc:  # pragma: no cover - depends on runtime availability
    pytest.skip(f"CuPy is installed but not usable in this environment: {exc}", allow_module_level=True)


def _build_context() -> tuple[Config, object, object, FFTManager]:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="cupy",
        dt_init=0.05,
        use_variable_dt=False,
        runtime_check_every=1,
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
    del kwargs
    return state.zeros_like()


def test_nonfinite_check_works_on_cupy_state() -> None:
    _, backend, grid, _ = _build_context()
    state = _finite_state(grid, backend)
    state["psi"][0, 0, 0] = complex(float("nan"), 0.0)

    with pytest.raises(NonFiniteStateError, match="psi"):
        check_state_finite(state, backend, time=0.0, step=0, context="gpu test")


def test_gpu_evolution_stops_on_nonfinite() -> None:
    config, backend, grid, fft = _build_context()
    state = _finite_state(grid, backend)
    call_count = {"n": 0}

    def bad_stepper(current: State, dt: float, ideal_rhs_func: object, linear_ops: object, rhs_kwargs=None) -> State:
        del dt, ideal_rhs_func, linear_ops, rhs_kwargs
        call_count["n"] += 1
        out = current.copy()
        if call_count["n"] >= 3:
            out["omega"][0, 0, 0] = complex(float("inf"), 0.0)
        return out

    with pytest.raises(NonFiniteStateError, match="omega"):
        evolve_until(
            state,
            0.2,
            _zero_rhs,
            _zero_linear_ops(state),
            rhs_kwargs={"grid": grid, "fft": fft, "workspace": None, "params": config, "dealias_mask": None},
            params=config,
            fixed_dt=0.05,
            stepper_func=bad_stepper,
        )
