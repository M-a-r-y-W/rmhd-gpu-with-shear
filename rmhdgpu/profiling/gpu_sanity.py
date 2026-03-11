"""Small CuPy sanity check for device residency, timing, and CPU/GPU agreement."""

from __future__ import annotations

import argparse
import importlib.util
from typing import Any

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.scalar import compute_energy_diagnostics, compute_scalar_diagnostics
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.profiling.benchmark_backends import run_backend_case
from rmhdgpu.state import State
from rmhdgpu.steppers import evolve_until
from rmhdgpu.workspace import Workspace


def _device_info(cp: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cupy_version": getattr(cp, "__version__", "unknown"),
    }
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        info.update(
            {
                "device_id": int(device.id),
                "device_name": name,
                "compute_capability": f"{props.get('major', '?')}.{props.get('minor', '?')}",
            }
        )
    except Exception as exc:  # pragma: no cover - best effort reporting
        info["device_query_error"] = str(exc)
    return info


def _deterministic_state(backend: Any, grid: Any, fft: FFTManager, mask: Any) -> State:
    state = State(grid, backend, field_names=s09.FIELD_NAMES)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)

    phi_real = (0.30 * xp.cos(x + z) + 0.12 * xp.sin(2.0 * y + z)).astype(
        grid.real_dtype,
        copy=False,
    )
    psi_real = (0.22 * xp.cos(x + y + z) + 0.09 * xp.sin(2.0 * x + z)).astype(
        grid.real_dtype,
        copy=False,
    )
    upar_real = (0.03 * xp.cos(x + 2.0 * z)).astype(grid.real_dtype, copy=False)
    dbpar_real = (0.025 * xp.sin(y + z)).astype(grid.real_dtype, copy=False)
    entropy_real = (0.015 * xp.sin(x + y + z)).astype(grid.real_dtype, copy=False)

    phi_hat = fft.r2c(phi_real)
    state["psi"][...] = fft.r2c(psi_real) * mask
    state["omega"][...] = s09.lap_perp(phi_hat * mask, grid)
    state["upar"][...] = fft.r2c(upar_real) * mask
    state["dbpar"][...] = fft.r2c(dbpar_real) * mask
    state["s"][...] = fft.r2c(entropy_real) * mask
    return state


def _short_run(backend_name: str, nx: int, steps: int, dt: float) -> tuple[dict[str, float], State]:
    config = Config(
        Nx=nx,
        Ny=nx,
        Nz=nx,
        backend=backend_name,
        dt_init=dt,
        use_variable_dt=False,
    )
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = 1.5e-3
        config.dissipation[name]["nu_par"] = 2.5e-3
        config.dissipation[name]["n_perp"] = 2
        config.dissipation[name]["n_par"] = 1

    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = _deterministic_state(backend, grid, fft, mask)
    final_state, _ = evolve_until(
        state,
        steps * dt,
        s09.ideal_rhs,
        linear_ops,
        rhs_kwargs={
            "grid": grid,
            "fft": fft,
            "workspace": workspace,
            "params": config,
            "dealias_mask": mask,
        },
        params=config,
        fixed_dt=dt,
    )
    diagnostics = compute_scalar_diagnostics(final_state, grid, fft, backend, workspace=workspace)
    diagnostics.update(compute_energy_diagnostics(final_state, grid, fft, backend, workspace=workspace))
    return diagnostics, final_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    args = parser.parse_args()

    if importlib.util.find_spec("cupy") is None:
        raise SystemExit("CuPy is not installed in this environment.")

    import cupy as cp  # Imported lazily so the script skips cleanly when unavailable.

    print({"backend": "cupy", **_device_info(cp)})
    benchmark_result = run_backend_case("cupy", args.nx, steps=args.steps, dt=args.dt)
    print({"cupy_benchmark": benchmark_result})

    gpu_diagnostics, gpu_state = _short_run("cupy", args.nx, args.steps, args.dt)
    cpu_diagnostics, cpu_state = _short_run("numpy", args.nx, args.steps, args.dt)

    print(
        {
            "cupy_state_on_device": isinstance(gpu_state["psi"], cp.ndarray),
            "numpy_state_on_device": False,
        }
    )

    differences = {}
    for name in gpu_state.field_names:
        gpu_field = cp.asnumpy(gpu_state[name])
        cpu_field = cpu_state.backend.to_numpy(cpu_state[name])
        diff = np.linalg.norm(gpu_field - cpu_field)
        ref = np.linalg.norm(cpu_field)
        differences[name] = float(diff / ref) if ref > 0.0 else float(diff)

    print({"cupy_diagnostics": gpu_diagnostics})
    print({"numpy_diagnostics": cpu_diagnostics})
    print({"relative_field_differences": differences})


if __name__ == "__main__":
    main()
