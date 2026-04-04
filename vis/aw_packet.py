"""Plot the Alfvén-wave packet profile for a `.input` case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.example_setups import build_initial_state
from rmhdgpu.examples.frame_output import (
    add_frame_arguments,
    build_frame_times,
    capture_xy_signed_fields,
    resolve_snapshot_z_index,
    write_signed_xy_frames,
)
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.runfile import resolve_run_settings
from rmhdgpu.steppers import if_ssprk3_step
from rmhdgpu.workspace import Workspace
from rmhdgpu.equations import s09


def _resolve_output_dir(settings: object, requested: str | None) -> Path:
    if requested is None:
        return settings.output_dir
    output_dir = Path(requested).expanduser()
    if output_dir.is_absolute():
        return output_dir
    base = settings.input_file.parent if settings.input_file is not None else Path.cwd()
    return (base / output_dir).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Path to a TOML-based .input file.")
    parser.add_argument("--output-dir", default=None, help="Directory where figures are written.")
    add_frame_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = resolve_run_settings(runfile_path=args.input_file)
    output_dir = _resolve_output_dir(settings, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = settings.config
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = build_initial_state(
        settings.initial_condition,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    z_index = resolve_snapshot_z_index(grid, args.snapshot_z_index)

    tau_A = grid.Lz / config.vA
    t_stop = min(config.tmax, 0.5 * tau_A)
    sample_times = np.arange(0.0, t_stop + 1.0e-15, 0.1 * tau_A)
    frame_times = build_frame_times(float(t_stop), args.frame_count) if args.save_frames else np.empty(0, dtype=np.float64)
    samples: list[tuple[float, np.ndarray]] = []
    frame_records: list[dict[str, object]] = []
    slice_x = grid.Nx // 3
    slice_y = grid.Ny // 4

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    t = 0.0
    sample_index = 0
    frame_index = 0
    while sample_index < len(sample_times) or frame_index < len(frame_times):
        next_profile_time = sample_times[sample_index] if sample_index < len(sample_times) else np.inf
        next_frame_time = frame_times[frame_index] if frame_index < len(frame_times) else np.inf
        target_time = min(next_profile_time, next_frame_time)

        if t >= target_time - 1.0e-15:
            if sample_index < len(sample_times) and next_profile_time <= target_time + 1.0e-15:
                psi_real = backend.to_numpy(fft.c2r(state["psi"]))
                samples.append((sample_times[sample_index] / tau_A, psi_real[slice_x, slice_y, :].copy()))
                sample_index += 1
            if frame_index < len(frame_times) and next_frame_time <= target_time + 1.0e-15:
                frame_records.append(
                    capture_xy_signed_fields(
                        state,
                        time=frame_times[frame_index],
                        grid=grid,
                        fft=fft,
                        backend=backend,
                        z_index=z_index,
                    )
                )
                frame_index += 1
            continue

        dt = min(config.dt_init, target_time - t)
        state = if_ssprk3_step(state, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
        t += dt

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(samples)))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    z = backend.to_numpy(grid.z)
    for color, (time_tau_A, profile) in zip(colors, samples, strict=True):
        ax.plot(z, profile, color=color, lw=2, label=f"t/tA={time_tau_A:.1f}")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\psi(x_0, y_0, z)$")
    ax.set_title("Exact nonlinear Alfvén-wave packet translation")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    figure_path = output_dir / "aw_packet_profile.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")
    write_signed_xy_frames(frame_records, output_dir=output_dir, grid=grid, z_index=z_index)


if __name__ == "__main__":
    main()
