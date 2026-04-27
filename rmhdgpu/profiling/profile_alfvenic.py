"""Profile the two-field Alfvenic timestep path.

This is a lightweight timing/memory probe rather than a formal test. It builds
a deterministic low-mode Alfvenic state, warms up one timestep so cuFFT plans
and RK scratch states are allocated, then reports steady-step timing, FFT call
counts, and CuPy memory-pool usage when running on a GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from typing import Any

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import alfvenic
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.state import State
from rmhdgpu.steppers import if_ssprk3_step
from rmhdgpu.workspace import Workspace


class CountingFFTManager:
    """Wrap FFTManager and count transform calls."""

    def __init__(self, fft: FFTManager) -> None:
        self._fft = fft
        self.grid = fft.grid
        self.backend = fft.backend
        self.xp = fft.xp
        self.r2c_calls = 0
        self.c2r_calls = 0

    def reset_counts(self) -> None:
        self.r2c_calls = 0
        self.c2r_calls = 0

    def r2c(self, f_real: Any, out: Any | None = None) -> Any:
        self.r2c_calls += 1
        return self._fft.r2c(f_real, out=out)

    def c2r(self, f_hat: Any, out: Any | None = None) -> Any:
        self.c2r_calls += 1
        return self._fft.c2r(f_hat, out=out)


def _maybe_clear_cupy_memory(backend: Any) -> None:
    if not backend.is_gpu:
        return

    backend.synchronize()
    gc.collect()
    xp = backend.xp
    try:
        xp.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    xp.get_default_memory_pool().free_all_blocks()
    xp.get_default_pinned_memory_pool().free_all_blocks()
    backend.synchronize()


def _gpu_memory_info(backend: Any) -> dict[str, float | str]:
    if not backend.is_gpu:
        return {}

    xp = backend.xp
    free, total = xp.cuda.runtime.memGetInfo()
    pool = xp.get_default_memory_pool()
    device = xp.cuda.Device()
    props = xp.cuda.runtime.getDeviceProperties(device.id)
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "device_name": name,
        "device_free_GiB": free / 1024**3,
        "device_total_GiB": total / 1024**3,
        "pool_used_GiB": pool.used_bytes() / 1024**3,
        "pool_total_GiB": pool.total_bytes() / 1024**3,
    }


def _build_config(backend_name: str, nx: int, dt: float) -> Config:
    return Config(
        equation_set="alfvenic",
        Nx=nx,
        Ny=nx,
        Nz=nx,
        backend=backend_name,
        dt_init=dt,
        use_variable_dt=False,
        use_forcing=False,
        progress_output_every=None,
    )


def _seed_low_mode_state(state: State, mask: Any) -> State:
    grid = state.grid
    entries = [
        ("psi", 1, 1, 1, 1.0 + 0.25j),
        ("psi", 2, 1, 1, -0.5 + 0.10j),
        ("psi", 1, 2, 2, 0.25 - 0.15j),
        ("omega", 1, 1, 1, 0.7 - 0.20j),
        ("omega", 1, 2, 1, -0.3 + 0.40j),
        ("omega", 2, 1, 2, 0.2 + 0.30j),
    ]
    for field_name, kx, ky, kz, value in entries:
        if kz < grid.fourier_shape[2]:
            state[field_name][kx % grid.Nx, ky % grid.Ny, kz] = value
    if mask is not None:
        state.apply_mask(mask)
    return state


def profile_alfvenic(
    backend_name: str,
    nx: int,
    *,
    dt: float,
    steps: int,
) -> dict[str, Any]:
    config = _build_config(backend_name, nx, dt)
    backend = build_backend(config)
    _maybe_clear_cupy_memory(backend)

    grid = build_grid(config, backend)
    fft_base = FFTManager(grid, backend)
    fft = CountingFFTManager(fft_base)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend, mode=config.dealias_mode) if config.dealias else None
    state = _seed_low_mode_state(State(grid, backend, field_names=alfvenic.FIELD_NAMES), mask)
    linear_ops = alfvenic.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    backend.synchronize()
    after_build = _gpu_memory_info(backend)

    current = if_ssprk3_step(state, dt, alfvenic.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
    backend.synchronize()
    after_warmup = _gpu_memory_info(backend)

    fft.reset_counts()
    start = time.perf_counter()
    for _ in range(steps):
        current = if_ssprk3_step(current, dt, alfvenic.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
    backend.synchronize()
    elapsed = time.perf_counter() - start
    after_timed = _gpu_memory_info(backend)

    result: dict[str, Any] = {
        "backend": backend_name,
        "nx": nx,
        "steps": steps,
        "dt": dt,
        "elapsed_s": elapsed,
        "time_per_step_s": elapsed / max(steps, 1),
        "fft_c2r_per_step": fft.c2r_calls / max(steps, 1),
        "fft_r2c_per_step": fft.r2c_calls / max(steps, 1),
        "fft_total_per_step": (fft.c2r_calls + fft.r2c_calls) / max(steps, 1),
    }
    if after_build:
        result["after_build"] = after_build
        result["after_warmup"] = after_warmup
        result["after_timed"] = after_timed
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cupy", choices=["numpy", "scipy_cpu", "cupy"])
    parser.add_argument("--nx", action="append", dest="sizes", type=int, help="Cubic grid size to profile.")
    parser.add_argument("--steps", type=int, default=4, help="Timed steps after one warmup step.")
    parser.add_argument("--dt", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sizes = args.sizes if args.sizes is not None else [128, 256]
    for nx in sizes:
        result = profile_alfvenic(args.backend, nx, dt=args.dt, steps=args.steps)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
