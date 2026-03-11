"""Qualitative decaying-turbulence spectra sanity check.

Run with:

`python -m rmhdgpu.examples.sanity_decay_spectra`

or, for a quick single-GPU `256^3` check,

`python -m rmhdgpu.examples.sanity_decay_spectra --gpu-256`

The script performs a modest dissipative run from low-mode initial data and
saves perpendicular shell spectra at several times.
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
from rmhdgpu.diagnostics.spectra import perpendicular_energy_spectrum_from_state
from rmhdgpu.equations import s09
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
from rmhdgpu.operators import dx, dy, lap_perp
from rmhdgpu.state import State
from rmhdgpu.steppers import compute_cfl_timestep, if_ssprk3_step
from rmhdgpu.workspace import Workspace


DEFAULT_BACKEND = "scipy_cpu"
DEFAULT_GRID_SIZE = 128
DEFAULT_T_FINAL = 4.0
DEFAULT_FFT_WORKERS = 8
GPU_256_T_FINAL = 1.0


def estimate_hyperdiffusion_coefficient(k_d: float, k0: float, u_rms: float, order: int) -> float:
    """Estimate `nu` from `tau_nl^{-1}(k_d) ~ nu * k_d^(2 n)`.

    The nonlinear rate estimate used is

    `tau_nl^{-1}(k_d) ~ k_d * u_rms * (k_d / k0)^(-1/3)`

    so the coefficient is chosen as

    `nu ~ k_d * u_rms * (k_d / k0)^(-1/3) / k_d^(2 n)`.
    """

    return k_d * u_rms * (k_d / k0) ** (-1.0 / 3.0) / (k_d ** (2 * order))


def _low_mode_real_field(
    grid: object,
    backend: object,
    seed: int,
    amplitude: float,
) -> object:
    rng = np.random.default_rng(seed)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = backend.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                a_cos = rng.normal(scale=amplitude / 6.0)
                a_sin = rng.normal(scale=amplitude / 6.0)
                phase = nx * x + ny * y + nz * z
                field += a_cos * xp.cos(phase) + a_sin * xp.sin(phase)

    return field.astype(grid.real_dtype, copy=False)


def _initial_u_rms(phi_hat: object, grid: object, fft: FFTManager, backend: object) -> float:
    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    xp = backend.xp
    return backend.scalar_to_float(xp.sqrt(xp.mean(ux**2 + uy**2)))


def dealiased_max_kperp(grid: object, backend: object, mask: object) -> float:
    """Return the maximum retained perpendicular wavenumber after 2/3 dealiasing."""

    kperp = np.sqrt(backend.to_numpy(grid.kperp2))
    retained = backend.to_numpy(mask).astype(bool)
    return float(np.max(kperp[retained]))


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
            ymax = max(ymax, positive.max())

        reference_k = spectra_by_time[0][1]["kperp"]
        mask = reference_k > 1.0
        if np.any(mask):
            k_ref = reference_k[mask]
            y_ref = 1.0e-3 * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
            axis.loglog(k_ref, y_ref, "k--", alpha=0.5, label=r"$k^{-5/3}$")

        mask0 = spectra_by_time[0][1]["kperp"] > 0.0
        y0 = spectra_by_time[0][1][key][mask0]
        positive0 = y0[y0 > 0.0]
        if positive0.size:
            ymax = positive0.max()
            axis.set_ylim(ymax * 1e-6, ymax)

        axis.set_title(titles[key])
        axis.set_xlabel(r"$k_\perp$")
        axis.grid(True, alpha=0.25)

    axes_flat[-1].axis("off")
    axes_flat[0].legend(fontsize=8)

    figure_path = output_dir / "sanity_decay_spectra.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved {figure_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="sanity_plots", help="Directory where figures are written.")
    parser.add_argument("--backend", choices=["numpy", "scipy_cpu", "cupy"], default=DEFAULT_BACKEND)
    parser.add_argument("--fft-workers", type=int, default=DEFAULT_FFT_WORKERS)
    parser.add_argument("--n", type=int, default=DEFAULT_GRID_SIZE, help="Grid resolution in each direction.")
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL, help="Final time.")
    parser.add_argument(
        "--gpu-256",
        action="store_true",
        help="Shortcut for backend='cupy', n=256, and a shorter quick-check runtime.",
    )
    add_frame_arguments(parser)
    return parser


def resolve_run_parameters(args: argparse.Namespace) -> dict[str, object]:
    """Resolve backend/grid defaults after CLI presets are applied."""

    backend_name = args.backend
    n = args.n
    fft_workers = args.fft_workers
    t_final = args.t_final
    if args.gpu_256:
        backend_name = "cupy"
        n = 256
        fft_workers = None
        if args.t_final == DEFAULT_T_FINAL:
            t_final = GPU_256_T_FINAL
    return {
        "backend_name": backend_name,
        "n": n,
        "fft_workers": fft_workers,
        "t_final": t_final,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_params = resolve_run_parameters(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        Nx=run_params["n"],
        Ny=run_params["n"],
        Nz=run_params["n"],
        backend=run_params["backend_name"],
        fft_workers=run_params["fft_workers"],
        cfl_number=0.5,
        dt_max=2.0e-2,
        tmax=run_params["t_final"],
        use_variable_dt=True,
    )

    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend)
    z_index = resolve_snapshot_z_index(grid, args.snapshot_z_index)

    print(
        "sanity_decay_spectra setup",
        {
            "backend": config.backend,
            "grid": grid.real_shape,
            "fft_workers": config.fft_workers,
            "t_final": config.tmax,
            "gpu_256": args.gpu_256,
            "save_frames": args.save_frames,
            "snapshot_z_index": z_index,
        },
    )

    state = State(grid, backend, field_names=config.field_names)
    phi_hat = fft.r2c(_low_mode_real_field(grid, backend, seed=1, amplitude=0.4)) * mask
    psi_hat = fft.r2c(_low_mode_real_field(grid, backend, seed=2, amplitude=0.3)) * mask
    state["psi"][...] = psi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    state["upar"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=3, amplitude=0.08)) * mask
    state["dbpar"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=4, amplitude=0.06)) * mask
    state["s"][...] = fft.r2c(_low_mode_real_field(grid, backend, seed=5, amplitude=0.05)) * mask

    u_rms = _initial_u_rms(phi_hat, grid, fft, backend)
    k0 = 1.0
    kperp_max_dealiased = dealiased_max_kperp(grid, backend, mask)
    # Choose the dissipation scale at roughly half the dealiased perpendicular
    # maximum so the damping acts near the top of the retained inertial range
    # rather than all the way at the truncation boundary.
    k_d = 0.7 * kperp_max_dealiased
    nu_perp = estimate_hyperdiffusion_coefficient(k_d=k_d, k0=k0, u_rms=u_rms, order=3)
    nu_par = 0.1 * nu_perp
    print(
        "dissipation-scale setup",
        {
            "k0": k0,
            "kperp_max_dealiased": kperp_max_dealiased,
            "k_d": k_d,
            "nu_perp": nu_perp,
            "nu_par": nu_par,
        },
    )
    for name in config.field_names:
        config.dissipation[name]["nu_perp"] = nu_perp
        config.dissipation[name]["nu_par"] = nu_par
        config.dissipation[name]["n_perp"] = 3
        config.dissipation[name]["n_par"] = 1

    linear_ops = s09.build_dissipation_operators(grid, config)
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

    current = state
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
