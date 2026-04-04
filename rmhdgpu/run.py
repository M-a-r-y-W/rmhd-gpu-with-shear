"""Main simulation driver for CLI and `.input` (TOML) workflows."""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rmhdgpu.backend import build_backend
from rmhdgpu.diagnostics.alfvenic import alfvenic_cross_helicity, alfvenic_energy
from rmhdgpu.diagnostics.scalar import compute_energy_diagnostics, compute_scalar_diagnostics
from rmhdgpu.errors import NonFiniteStateError
from rmhdgpu.example_setups import build_initial_state
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.grid import build_grid
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.runfile import (
    LEGACY_INPUT_SUFFIX,
    PRIMARY_INPUT_SUFFIX,
    RunSettings,
    cli_overrides_from_args,
    resolve_run_settings,
    write_resolved_config,
)
from rmhdgpu.steppers import compute_cfl_timestep, if_ssprk3_step
from rmhdgpu.utils import check_state_finite
from rmhdgpu.workspace import Workspace


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class _RunLogger:
    path: Path
    _handle: Any

    def event(self, label: str, payload: dict[str, Any] | None = None) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        if payload is None:
            line = f"{timestamp} {label}"
        else:
            line = f"{timestamp} {label} {payload}"
        print(line)
        print(line, file=self._handle)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for normal runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help=(
            f"Optional {PRIMARY_INPUT_SUFFIX} input file. These files use TOML syntax internally. "
            f"Legacy {LEGACY_INPUT_SUFFIX} files are also accepted."
        ),
    )
    parser.add_argument("--title", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)

    parser.add_argument("--backend", choices=["numpy", "scipy_cpu", "cupy"], default=argparse.SUPPRESS)
    parser.add_argument("--fft-workers", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--nx", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ny", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nz", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lx", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--ly", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--lz", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--tmax", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-init", dest="dt_init", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-min", dest="dt_min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt-max", dest="dt_max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--cfl-number", dest="cfl_number", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-scal", dest="t_out_scal", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-spec", dest="t_out_spec", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--t-out-full", dest="t_out_full", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--use-variable-dt", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--runtime-check-every", dest="runtime_check_every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--progress-output-every", dest="progress_output_every", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fail-on-nonfinite", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--dealias", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--dealias-mode", default=argparse.SUPPRESS)

    parser.add_argument("--vA", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--cs2-over-vA2", dest="cs2_over_vA2", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--use-forcing", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--force-sigma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--forcing-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-min-force", dest="n_min_force", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--n-max-force", dest="n_max_force", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--alpha-force", dest="alpha_force", type=float, default=argparse.SUPPRESS)

    parser.add_argument(
        "--initial-condition",
        choices=["alfven_mode", "zero"],
        default=argparse.SUPPRESS,
        help="Initial condition family for CLI mode or run-file override.",
    )
    parser.add_argument("--mode-kx", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-ky", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-kz", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode-amplitude", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--mode-branch", choices=["plus", "minus"], default=argparse.SUPPRESS)
    return parser


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _prepare_output_dir(settings: RunSettings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir


def _open_run_logger(output_dir: Path) -> _RunLogger:
    log_path = output_dir / "run.log"
    handle = log_path.open("w", encoding="utf-8")
    return _RunLogger(path=log_path, _handle=handle)


def _write_input_copy(settings: RunSettings, output_dir: Path) -> None:
    if settings.input_file is None:
        return
    shutil.copyfile(settings.input_file, output_dir / "input_copy.input")


def _startup_metadata(settings: RunSettings, output_dir: Path) -> dict[str, Any]:
    return {
        "title": settings.title,
        "input_file": None if settings.input_file is None else str(settings.input_file),
        "output_dir": str(output_dir),
        "backend": settings.config.backend,
        "grid": [settings.config.Nx, settings.config.Ny, settings.config.Nz],
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "git_commit": _git_commit_hash(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
    }


def _diagnostics_row(
    state: State,
    *,
    step: int,
    time: float,
    dt: float,
    grid: Any,
    fft: Any,
    backend: Any,
    workspace: Any,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": step,
        "t": time,
        "dt": dt,
    }
    row.update(compute_scalar_diagnostics(state, grid, fft, backend, workspace=workspace))
    row.update(compute_energy_diagnostics(state, grid, fft, backend, workspace=workspace))
    row["alfvenic_cross_helicity"] = alfvenic_cross_helicity(state, grid, fft)
    return row


def _write_scalar_row(
    writer: csv.DictWriter[str] | None,
    handle: Any,
    row: dict[str, float | int],
) -> csv.DictWriter[str]:
    if writer is None:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
    writer.writerow(row)
    handle.flush()
    return writer


def run_simulation(settings: RunSettings) -> dict[str, Any]:
    """Run one resolved case and write standard outputs."""

    config = settings.config
    output_dir = _prepare_output_dir(settings)
    logger = _open_run_logger(output_dir)
    try:
        write_resolved_config(settings, output_dir / "resolved_config.toml")
        _write_input_copy(settings, output_dir)

        logger.event("run start", _startup_metadata(settings, output_dir))
        logger.event(
            "run setup",
            {
                "resolved_output_dir": str(output_dir),
                "input_file": None if settings.input_file is None else str(settings.input_file),
                "backend": config.backend,
                "grid": [config.Nx, config.Ny, config.Nz],
                "initial_condition": settings.initial_condition.to_document(),
            },
        )

        backend = build_backend(config)
        grid = build_grid(config, backend)
        fft = FFTManager(grid, backend)
        mask = build_dealias_mask(grid, backend, mode=config.dealias_mode) if config.dealias else None
        workspace = Workspace(grid, backend)
        linear_ops = s09.build_dissipation_operators(grid, config)
        state = build_initial_state(
            settings.initial_condition,
            grid=grid,
            backend=backend,
            fft=fft,
            dealias_mask=mask,
            field_names=settings.config.field_names,
            params=settings.config,
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

        scalar_csv_path = output_dir / "scalar_diagnostics.csv"
        with scalar_csv_path.open("w", encoding="utf-8", newline="") as scalar_handle:
            scalar_writer: csv.DictWriter[str] | None = None

            t = 0.0
            steps = 0
            next_scalar_output = config.t_out_scal
            dt_last = config.dt_init
            forcing_rng = backend.random_generator(config.forcing_seed) if config.use_forcing else None

            try:
                if config.fail_on_nonfinite:
                    check_state_finite(state, backend, time=t, step=steps, context="run startup")

                initial_row = _diagnostics_row(
                    state,
                    step=steps,
                    time=t,
                    dt=0.0,
                    grid=grid,
                    fft=fft,
                    backend=backend,
                    workspace=workspace,
                )
                scalar_writer = _write_scalar_row(scalar_writer, scalar_handle, initial_row)
                logger.event("scalar diagnostics", initial_row)

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
                        logger.event("progress", {"step": steps, "t": t, "dt": dt})

                    if config.fail_on_nonfinite and (
                        steps % config.runtime_check_every == 0 or t >= config.tmax - 1.0e-15
                    ):
                        check_state_finite(state, backend, time=t, step=steps, context="run")

                    if t >= next_scalar_output - 1.0e-15 or t >= config.tmax - 1.0e-15:
                        row = _diagnostics_row(
                            state,
                            step=steps,
                            time=t,
                            dt=dt,
                            grid=grid,
                            fft=fft,
                            backend=backend,
                            workspace=workspace,
                        )
                        scalar_writer = _write_scalar_row(scalar_writer, scalar_handle, row)
                        logger.event("scalar diagnostics", row)
                        next_scalar_output += config.t_out_scal
            except NonFiniteStateError as exc:
                logger.event("run failed", {"error": str(exc)})
                raise SystemExit(str(exc)) from exc

        energy_final = alfvenic_energy(state, grid, fft)
        cross_final = alfvenic_cross_helicity(state, grid, fft)
        summary = {
            "backend": config.backend,
            "grid": list(grid.real_shape),
            "steps": steps,
            "t_final": t,
            "dt_last": dt_last,
            "alfvenic_energy_initial": energy_initial,
            "alfvenic_energy_final": energy_final,
            "alfvenic_cross_helicity_initial": cross_initial,
            "alfvenic_cross_helicity_final": cross_final,
            "psi_hat_max_abs": backend.scalar_to_float(backend.xp.max(backend.xp.abs(state["psi"]))),
            "output_dir": str(output_dir),
            "scalar_diagnostics_csv": str(scalar_csv_path),
        }
        logger.event("run complete", summary)
        return summary
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> None:
    """Resolve CLI or `.input` input and execute one simulation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_file is not None and not str(args.input_file).endswith((PRIMARY_INPUT_SUFFIX, LEGACY_INPUT_SUFFIX)):
        raise SystemExit(
            "Expected an optional input file with suffix "
            f"{PRIMARY_INPUT_SUFFIX!r} (legacy {LEGACY_INPUT_SUFFIX!r} is also accepted) "
            f"as the first positional argument; got {args.input_file!r}."
        )

    try:
        settings = resolve_run_settings(
            runfile_path=args.input_file,
            cli_overrides=cli_overrides_from_args(args),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_simulation(settings)


if __name__ == "__main__":
    main()
