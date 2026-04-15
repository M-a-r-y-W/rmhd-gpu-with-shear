"""Qualitative sanity check for anisotropic single-mode dissipation.

Run with:

`python -m rmhdgpu.examples.sanity_single_mode_decay`

The script saves a plot comparing numerical and exact exponential decay for a
perpendicular-dominated and a parallel-dominated damping case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.testing import single_mode_field
from rmhdgpu.state import State
from rmhdgpu.steppers import if_ssprk3_step


def _zero_rhs(state: State, **kwargs) -> State:
    return state.zeros_like()


def _run_case(
    field_name: str,
    mode_indices: tuple[int, int, int],
    dissipation: dict[str, float | int],
    total_time: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = Config(Nx=16, Ny=16, Nz=16, backend="numpy", use_variable_dt=False)
    config.dissipation[field_name].update(dissipation)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)

    state = State(grid, backend, field_names=config.field_names)
    state[field_name][...] = single_mode_field(grid, backend, mode_indices, amplitude=1.0)
    amplitude0 = complex(backend.to_numpy(state[field_name])[mode_indices])
    damping = float(backend.to_numpy(linear_ops[field_name])[mode_indices])

    times = [0.0]
    amplitudes = [abs(amplitude0)]
    exact = [abs(amplitude0)]
    current = state

    steps = int(round(total_time / dt))
    for step in range(1, steps + 1):
        current = if_ssprk3_step(current, dt, _zero_rhs, linear_ops)
        time = step * dt
        amplitude = complex(backend.to_numpy(current[field_name])[mode_indices])
        times.append(time)
        amplitudes.append(abs(amplitude))
        exact.append(abs(amplitude0) * np.exp(-damping * time))

    return np.asarray(times), np.asarray(amplitudes), np.asarray(exact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_plots", help="Directory where figures are written.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "Perpendicular-dominated",
            "psi",
            (2, 1, 1),
            {"nu_perp": 5.0e-3, "nu_par": 0.0, "n_perp": 2, "n_par": 1},
        ),
        (
            "Parallel-dominated",
            "upar",
            (1, 1, 3),
            {"nu_perp": 0.0, "nu_par": 5.0e-2, "n_perp": 2, "n_par": 1},
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (title, field_name, mode_indices, dissipation) in zip(axes, cases, strict=True):
        times, amplitudes, exact = _run_case(field_name, mode_indices, dissipation, total_time=0.6, dt=0.05)
        axis.plot(times, amplitudes, label="Numerical", lw=2)
        axis.plot(times, exact, "--", label="Exact", lw=2)
        axis.set_title(title)
        axis.set_xlabel("t")
        axis.set_ylabel("|q_hat|")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure_path = output_dir / "sanity_single_mode_decay.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
