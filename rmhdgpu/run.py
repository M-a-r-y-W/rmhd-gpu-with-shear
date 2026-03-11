"""Small smoke driver for the dissipative homogeneous five-field prototype."""

from __future__ import annotations

import argparse

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.alfvenic import alfvenic_cross_helicity, alfvenic_energy
from rmhdgpu.diagnostics.scalar import compute_energy_diagnostics, compute_scalar_diagnostics
from rmhdgpu.errors import NonFiniteStateError
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds.eigenmodes import alfven_mode_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.steppers import _print_progress, compute_cfl_timestep, if_ssprk3_step
from rmhdgpu.utils import check_state_finite
from rmhdgpu.workspace import Workspace


def main() -> None:
    """Run a tiny end-to-end dissipative s09 setup, optionally with forcing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy", choices=["numpy", "scipy_cpu", "cupy"])
    parser.add_argument("--fft-workers", type=int, default=None)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--tmax", type=float, default=5.0e-2)
    parser.add_argument("--use-forcing", action="store_true")
    parser.add_argument("--force-sigma", type=float, default=5.0e-2)
    parser.add_argument("--forcing-seed", type=int, default=1234)
    parser.add_argument("--progress-output-every", type=int, default=100)
    args = parser.parse_args()

    config = Config(
        Nx=args.nx,
        Ny=args.nx,
        Nz=args.nx,
        backend=args.backend,
        fft_workers=args.fft_workers,
        dt_init=5.0e-3,
        dt_max=2.0e-2,
        tmax=args.tmax,
        use_forcing=args.use_forcing,
        forcing_seed=args.forcing_seed,
        progress_output_every=args.progress_output_every,
    )
    config.dissipation["psi"]["nu_perp"] = 5.0e-3
    config.dissipation["omega"]["nu_perp"] = 5.0e-3
    if config.use_forcing:
        config.force_amplitudes["psi"] = args.force_sigma
        config.force_amplitudes["omega"] = args.force_sigma
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    mask = build_dealias_mask(grid, backend, mode=config.dealias_mode) if config.dealias else None
    workspace = Workspace(grid, backend)
    linear_ops = s09.build_dissipation_operators(grid, config)
    state = alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=config.field_names,
        k_indices=(1, 1, 1),
        amplitude=1.0,
        branch="plus",
        params=config,
    )

    energy_initial = alfvenic_energy(state, grid, fft)
    cross_initial = alfvenic_cross_helicity(state, grid, fft)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    t = 0.0
    steps = 0
    next_scalar_output = 0.0
    dt_last = config.dt_init
    forcing_rng = backend.random_generator(config.forcing_seed) if config.use_forcing else None
    print(
        "backend configuration",
        {
            "backend": backend.backend_name,
            "fft_workers": backend.fft_workers,
            "use_forcing": config.use_forcing,
            "forcing_seed": config.forcing_seed if config.use_forcing else None,
        },
    )

    try:
        if config.fail_on_nonfinite:
            check_state_finite(state, backend, time=t, step=steps, context="smoke run startup")

        while t < config.tmax - 1.0e-15:
            if config.use_variable_dt:
                dt = compute_cfl_timestep(state, grid, fft, config, dt_prev=dt_last, workspace=workspace)
            else:
                dt = config.dt_init
            dt = min(dt, config.tmax - t)
            state = if_ssprk3_step(state, dt, s09.ideal_rhs, linear_ops, rhs_kwargs=rhs_kwargs)
            if config.use_forcing:
                forcing_kick = generate_forcing_kick(
                    state,
                    grid,
                    fft,
                    backend,
                    config,
                    forcing_rng,
                    dt,
                    workspace=workspace,
                    out=workspace.get_state_buffer("forcing_kick", state.field_names),
                )
                state = apply_forcing_kick(state, forcing_kick, inplace=True)
            t += dt
            dt_last = dt
            steps += 1

            if config.progress_output_every is not None and (
                steps % config.progress_output_every == 0 or t >= config.tmax - 1.0e-15
            ):
                _print_progress(steps, t, dt)

            if config.fail_on_nonfinite and (
                steps % config.runtime_check_every == 0 or t >= config.tmax - 1.0e-15
            ):
                check_state_finite(state, backend, time=t, step=steps, context="smoke run")

            if t >= next_scalar_output - 1.0e-15 or t >= config.tmax - 1.0e-15:
                diagnostics = compute_scalar_diagnostics(state, grid, fft, backend, workspace=workspace)
                print(
                    "scalar diagnostics",
                    {
                        "t": t,
                        "dt": dt,
                        "psi_rms": diagnostics["psi_rms"],
                        **compute_energy_diagnostics(state, grid, fft, backend, workspace=workspace),
                        "alfvenic_cross_helicity": alfvenic_cross_helicity(state, grid, fft),
                    },
                )
                next_scalar_output += config.t_out_scal
    except NonFiniteStateError as exc:
        raise SystemExit(str(exc)) from exc

    energy_final = alfvenic_energy(state, grid, fft)
    cross_final = alfvenic_cross_helicity(state, grid, fft)
    print(
        "rmhdgpu smoke example",
        {
            "backend": config.backend,
            "grid": grid.real_shape,
            "steps": steps,
            "alfvenic_energy_initial": energy_initial,
            "alfvenic_energy_final": energy_final,
            "alfvenic_cross_helicity_initial": cross_initial,
            "alfvenic_cross_helicity_final": cross_final,
            "t_final": t,
            "dt_last": dt_last,
            "psi_hat_max_abs": backend.scalar_to_float(backend.xp.max(backend.xp.abs(state["psi"]))),
        },
    )


if __name__ == "__main__":
    main()
