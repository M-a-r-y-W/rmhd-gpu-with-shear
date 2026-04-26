from __future__ import annotations

import csv
from pathlib import Path

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.diagnostics.scalar import compute_scalar_diagnostics
from rmhdgpu.equations import alfvenic, low_beta_stratified, s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.run import main
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


def _build_context(config: Config):
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    state = State(grid, backend, field_names=config.field_names)
    return state, backend, grid, fft, workspace


def test_generic_field_scalar_diagnostics_exist() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8)
    state, backend, grid, fft, workspace = _build_context(config)

    diagnostics = compute_scalar_diagnostics(state, grid, fft, backend, workspace=workspace)

    for field_name in config.field_names:
        assert f"{field_name}_mean" in diagnostics
        assert f"{field_name}_rms" in diagnostics
        assert f"{field_name}_max_abs" in diagnostics


def test_s09_equation_scalar_diagnostics_include_expected_energy_names() -> None:
    config = Config(equation_set="s09", Nx=8, Ny=8, Nz=8)
    state, backend, grid, fft, workspace = _build_context(config)
    linear_ops = s09.build_dissipation_operators(grid, config)

    diagnostics = s09.compute_equation_scalar_diagnostics(
        state,
        grid=grid,
        fft=fft,
        backend=backend,
        params=config,
        workspace=workspace,
        linear_ops=linear_ops,
    )

    for name in (
        "total_energy",
        "total_energy_rhs_total",
        "total_energy_rhs_dissipation",
        "total_energy_rhs_forcing",
        "alfvenic_energy",
        "entropy_variance",
    ):
        assert name in diagnostics
        assert name in s09.SCALAR_DIAGNOSTIC_INFO


def test_alfvenic_equation_scalar_diagnostics_include_expected_energy_names() -> None:
    config = Config(equation_set="alfvenic", Nx=8, Ny=8, Nz=8)
    state, backend, grid, fft, workspace = _build_context(config)
    linear_ops = alfvenic.build_dissipation_operators(grid, config)

    diagnostics = alfvenic.compute_equation_scalar_diagnostics(
        state,
        grid=grid,
        fft=fft,
        backend=backend,
        params=config,
        workspace=workspace,
        linear_ops=linear_ops,
    )

    for name in (
        "total_energy",
        "total_energy_rhs_total",
        "total_energy_rhs_dissipation",
        "total_energy_rhs_forcing",
        "alfvenic_energy",
    ):
        assert name in diagnostics
        assert name in alfvenic.SCALAR_DIAGNOSTIC_INFO


def test_low_beta_equation_scalar_diagnostics_include_expected_energy_names() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8, N2=0.25)
    state, backend, grid, fft, workspace = _build_context(config)
    linear_ops = low_beta_stratified.build_dissipation_operators(grid, config)

    diagnostics = low_beta_stratified.compute_equation_scalar_diagnostics(
        state,
        grid=grid,
        fft=fft,
        backend=backend,
        params=config,
        workspace=workspace,
        linear_ops=linear_ops,
    )

    for name in (
        "total_energy",
        "total_energy_rhs_total",
        "total_energy_rhs_dissipation",
        "total_energy_rhs_forcing",
        "total_energy_rhs_stratification",
        "a_energy",
    ):
        assert name in diagnostics
        assert name in low_beta_stratified.SCALAR_DIAGNOSTIC_INFO


def test_scalar_csv_contains_generic_and_equation_specific_columns(tmp_path: Path) -> None:
    input_file = tmp_path / "scalar_columns.input"
    input_file.write_text(
        """
title = "Scalar column split test"
output_dir = "outputs"

[equations]
type = "low_beta_stratified"

[grid]
Nx = 8
Ny = 8
Nz = 8

[time]
tmax = 0.005
dt_init = 0.005
dt_max = 0.005
use_variable_dt = false

[output]
t_out_scal = 0.005
t_out_spec = 0.0
t_out_full = 0.0

[backend]
backend = "numpy"

[runtime]
progress_output_every = 100

[physics]
vA = 1.0
N2 = 0.25

[initial_condition]
type = "low_beta_stratified_mode"

[initial_condition.parameters]
k_indices = [0, 1, 0]
mode = "unstable_growing"
amplitude = 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )

    main([str(input_file)])

    csv_path = tmp_path / "outputs" / "scalar_diagnostics.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert rows
    assert "psi_rms" in reader.fieldnames
    assert "a_max_abs" in reader.fieldnames
    assert "total_energy" in reader.fieldnames
    assert "total_energy_rhs_total" in reader.fieldnames
    assert "total_energy_rhs_stratification" in reader.fieldnames
