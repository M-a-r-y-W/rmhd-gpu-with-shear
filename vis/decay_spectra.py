"""Plot perpendicular spectra for a decaying-turbulence `.input` case."""

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
from rmhdgpu.diagnostics.spectra import perpendicular_energy_spectrum_from_state
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
from rmhdgpu.steppers import compute_cfl_timestep, if_ssprk3_step
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


def _plot_spectra(
    spectra_by_time: list[tuple[float, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    keys = ["u_perp", "b_perp", "upar", "dbpar", "s"]
    titles = {
        "u_perp": r"$E_{u_\perp}(k_\perp)$",
        "b_perp": r"$E_{b_\perp}(k_\perp)$",
        "upar": r"$E_{u_\parallel}(k_\perp)$",
        "dbpar": r"$E_{\delta B_\parallel}(k_\perp)$",
        "s": r"$E_s(k_\perp)$",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = list(axes.flat)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(spectra_by_time)))

    for axis, key in zip(axes_flat[: len(keys)], keys, strict=True):
        ymax = 0.0
        for color, (time, spectra) in zip(colors, spectra_by_time, strict=True):
            mask = spectra["kperp"] > 0.0
            x = spectra["kperp"][mask]
            y = spectra[key][mask]
            axis.loglog(x, y, color=color, lw=2, label=f"t={time:.2f}")
            positive = y[y > 0.0]
            if positive.size:
                ymax = max(ymax, float(positive.max()))

        reference_k = spectra_by_time[0][1]["kperp"]
        mask = reference_k > 1.0
        if np.any(mask):
            k_ref = reference_k[mask]
            y_ref = 1.0e-3 * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
            axis.loglog(k_ref, y_ref, "k--", alpha=0.5, label=r"$k^{-5/3}$")

        if ymax > 0.0:
            axis.set_ylim(ymax * 1e-6, ymax)
        axis.set_title(titles[key])
        axis.set_xlabel(r"$k_\perp$")
        axis.grid(True, alpha=0.25)

    axes_flat[-1].axis("off")
    axes_flat[0].legend(fontsize=8)

    figure_path = output_dir / "decay_spectra.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


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
    current = build_initial_state(
        settings.initial_condition,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    z_index = resolve_snapshot_z_index(grid, args.snapshot_z_index)

    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }

    sample_times = np.linspace(0.0, config.tmax, 10)
    frame_times = build_frame_times(config.tmax, args.frame_count) if args.save_frames else np.empty(0, dtype=np.float64)
    spectra_by_time: list[tuple[float, dict[str, np.ndarray]]] = []
    frame_records: list[dict[str, object]] = []

    t = 0.0
    sample_index = 0
    frame_index = 0
    while sample_index < len(sample_times) or frame_index < len(frame_times):
        next_spectrum_time = sample_times[sample_index] if sample_index < len(sample_times) else np.inf
        next_frame_time = frame_times[frame_index] if frame_index < len(frame_times) else np.inf
        target_time = min(next_spectrum_time, next_frame_time)

        if t >= target_time - 1.0e-15:
            if sample_index < len(sample_times) and next_spectrum_time <= target_time + 1.0e-15:
                spectra_by_time.append(
                    (sample_times[sample_index], perpendicular_energy_spectrum_from_state(current, grid, backend))
                )
                sample_index += 1
            if frame_index < len(frame_times) and next_frame_time <= target_time + 1.0e-15:
                frame_records.append(
                    capture_xy_signed_fields(
                        current,
                        time=frame_times[frame_index],
                        grid=grid,
                        fft=fft,
                        backend=backend,
                        z_index=z_index,
                    )
                )
                frame_index += 1
            continue

        dt = compute_cfl_timestep(current, grid, fft, config, workspace=workspace)
        dt = min(dt, target_time - t)
        current = if_ssprk3_step(current, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
        t += dt

    _plot_spectra(spectra_by_time, output_dir)
    write_signed_xy_frames(frame_records, output_dir=output_dir, grid=grid, z_index=z_index)


if __name__ == "__main__":
    main()
