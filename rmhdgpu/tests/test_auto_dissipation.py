from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rmhdgpu.auto_dissipation import AutoDissipationController
from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.initconds.testing import random_spectrum_test_parameters
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.run import main
from rmhdgpu.runfile import resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


def _build_context(config: Config) -> tuple[object, object, FFTManager, Workspace, object]:
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend) if config.dealias else None
    return backend, grid, fft, workspace, mask


def _build_test_state(
    config: Config,
    *,
    rms: float = 0.1,
) -> tuple[object, object, State, object]:
    backend, grid, fft, _, mask = _build_context(config)
    state = build_initial_state(
        "random_spectrum",
        parameters=random_spectrum_test_parameters(rms),
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    return backend, grid, state, mask


def _write_auto_input(path: Path) -> None:
    path.write_text(
        """
title = "Auto dissipation smoke test"
output_dir = "outputs"

[grid]
Nx = 8
Ny = 8
Nz = 8

[time]
tmax = 0.02
dt_init = 0.005
dt_max = 0.005
use_variable_dt = false

[output]
t_out_scal = 0.01
t_out_spec = 0.0
t_out_full = 0.0

[backend]
backend = "numpy"

[runtime]
progress_output_every = 100

[initial_condition]
type = "random_spectrum"

[dissipation]
mode = "auto"
n_perp = 3
n_par = 1
nu_par = 0.0
kd_fraction = 0.6
shell_half_width = 0.5
update_every = 1
smooth_factor = 0.3
nu_min = 1e-12
nu_max = 1e-2
max_update_factor = 2.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_auto_dissipation_input_parses(tmp_path) -> None:
    input_file = tmp_path / "auto.input"
    _write_auto_input(input_file)

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.auto_dissipation.enabled is True
    assert settings.config.auto_dissipation.n_perp == 3
    assert settings.config.auto_dissipation.n_par == 1
    assert settings.config.auto_dissipation.nu_par == 0.0
    assert settings.config.auto_dissipation.kd_fraction == 0.6
    assert settings.config.auto_dissipation.update_every == 1


def test_manual_mode_still_works(tmp_path) -> None:
    input_file = tmp_path / "manual.input"
    input_file.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[dissipation]
mode = "manual"

[dissipation.psi]
nu_perp = 0.005
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.auto_dissipation.enabled is False
    assert settings.config.dissipation["psi"]["nu_perp"] == 0.005
    assert settings.config.dissipation["omega"]["nu_perp"] == 0.0


def test_auto_dissipation_update_returns_finite_values() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        auto_dissipation={"mode": "auto", "update_every": 1, "nu_max": 1.0},
    )
    backend, grid, state, mask = _build_test_state(config)
    controller = AutoDissipationController.from_runtime(
        settings=config.auto_dissipation,
        equation_module=s09,
        field_names=config.field_names,
        grid=grid,
        backend=backend,
        dealias_mask=mask,
    )

    nu_perp = controller.update(state, config)

    assert np.isfinite(nu_perp)
    assert np.isfinite(controller.last_ud)
    assert np.isfinite(controller.last_Ed)
    assert nu_perp > 0.0
    assert controller.last_ud > 0.0
    assert controller.last_Ed > 0.0


def test_auto_dissipation_no_fft_needed() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", auto_dissipation={"mode": "auto"})
    backend, grid, state, mask = _build_test_state(config)
    controller = AutoDissipationController.from_runtime(
        settings=config.auto_dissipation,
        equation_module=s09,
        field_names=config.field_names,
        grid=grid,
        backend=backend,
        dealias_mask=mask,
    )

    # The controller works directly from the Fourier-space state and the grid;
    # no FFT object is required for the update.
    nu_perp = controller.update(state, config)

    assert np.isfinite(nu_perp)


def test_auto_dissipation_smoothing_behaves_reasonably() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        auto_dissipation={
            "mode": "auto",
            "smooth_factor": 0.5,
            "nu_min": 1.0e-8,
            "nu_max": 1.0,
            "max_update_factor": 1.0e6,
        },
    )
    backend, grid, _, _, mask = _build_context(config)
    state = State(grid, backend, field_names=config.field_names)
    controller = AutoDissipationController.from_runtime(
        settings=config.auto_dissipation,
        equation_module=s09,
        field_names=config.field_names,
        grid=grid,
        backend=backend,
        dealias_mask=mask,
    )
    controller.current_nu_perp = 1.0e-4

    nu_perp = controller.update(state, config)

    assert np.isclose(nu_perp, 1.0e-6, rtol=1.0e-12, atol=1.0e-18)


def test_auto_mode_gives_common_coefficients_to_all_fields() -> None:
    config = Config(
        Nx=8,
        Ny=8,
        Nz=8,
        backend="numpy",
        auto_dissipation={"mode": "auto", "n_perp": 2, "n_par": 1, "nu_par": 0.03},
    )
    backend, grid, state, mask = _build_test_state(config)
    controller = AutoDissipationController.from_runtime(
        settings=config.auto_dissipation,
        equation_module=s09,
        field_names=config.field_names,
        grid=grid,
        backend=backend,
        dealias_mask=mask,
    )
    controller.update(state, config)
    effective = controller.effective_dissipation()
    linear_ops = s09.build_dissipation_operators(grid, config, dissipation_spec=effective)

    reference_spec = effective["psi"]
    for name in config.field_names:
        assert effective[name] == reference_spec
        np.testing.assert_allclose(
            backend.to_numpy(linear_ops[name]),
            backend.to_numpy(linear_ops["psi"]),
            atol=0.0,
            rtol=0.0,
        )


def test_manual_mode_preserves_per_field_coefficients() -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy")
    config.dissipation["psi"].update({"nu_perp": 0.01, "nu_par": 0.0, "n_perp": 2, "n_par": 1})
    config.dissipation["upar"].update({"nu_perp": 0.03, "nu_par": 0.0, "n_perp": 2, "n_par": 1})
    backend, grid, _, _, _ = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)

    psi_rate = backend.to_numpy(linear_ops["psi"])[1, 1, 1]
    upar_rate = backend.to_numpy(linear_ops["upar"])[1, 1, 1]

    assert upar_rate > psi_rate


def test_small_auto_dissipation_run_executes(tmp_path) -> None:
    input_file = tmp_path / "auto_run.input"
    _write_auto_input(input_file)

    main([str(input_file)])

    output_dir = tmp_path / "outputs"
    assert (output_dir / "resolved_config.toml").exists()
    assert (output_dir / "scalar_diagnostics.csv").exists()


def test_auto_dissipation_scalar_columns_written(tmp_path) -> None:
    input_file = tmp_path / "auto_columns.input"
    _write_auto_input(input_file)

    main([str(input_file)])

    csv_path = tmp_path / "outputs" / "scalar_diagnostics.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    for name in (
        "auto_dissipation_enabled",
        "auto_dissipation_nu_perp",
        "auto_dissipation_nu_par",
        "auto_dissipation_kd",
        "auto_dissipation_ud",
        "auto_dissipation_Ed",
    ):
        assert name in reader.fieldnames
    assert rows
