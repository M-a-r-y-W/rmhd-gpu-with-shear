from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import available_equation_sets, get_equation_module, low_beta_stratified, s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.initconds import build_initial_state
from rmhdgpu.masks import build_dealias_mask
from rmhdgpu.run import main
from rmhdgpu.runfile import resolve_run_settings
from rmhdgpu.state import State
from rmhdgpu.steppers import ssprk3_step
from rmhdgpu.workspace import Workspace
from vis.plot_budget import main as plot_budget_main


def _build_context(config: Config) -> tuple[object, object, FFTManager, Workspace, object]:
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)
    mask = build_dealias_mask(grid, backend) if config.dealias else None
    return backend, grid, fft, workspace, mask


def _select_low_beta_eigenvalue(eigenvalues: np.ndarray, mode: str) -> complex:
    if mode == "unstable_growing":
        return complex(eigenvalues[int(np.argmax(eigenvalues.real))])
    if mode == "unstable_decaying":
        return complex(eigenvalues[int(np.argmin(eigenvalues.real))])
    if mode == "stable_plus":
        return complex(eigenvalues[int(np.argmax(eigenvalues.imag))])
    if mode == "stable_minus":
        return complex(eigenvalues[int(np.argmin(eigenvalues.imag))])
    raise ValueError(mode)


def _low_beta_mode_state(
    config: Config,
    *,
    k_indices: tuple[int, int, int],
    mode: str,
    amplitude: float = 0.1,
) -> tuple[State, object, object, FFTManager, Workspace, object, complex]:
    backend, grid, fft, workspace, mask = _build_context(config)
    state = build_initial_state(
        "low_beta_stratified_mode",
        parameters={
            "k_indices": list(k_indices),
            "amplitude": amplitude,
            "mode": mode,
        },
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=mask,
        field_names=config.field_names,
        params=config,
    )
    ix, iy, iz = k_indices
    kx = backend.scalar_to_float(grid.kx[ix % grid.Nx, 0, 0])
    ky = backend.scalar_to_float(grid.ky[0, iy % grid.Ny, 0])
    kz = backend.scalar_to_float(grid.kz[0, 0, iz])
    eigenvalues = np.linalg.eigvals(low_beta_stratified.linear_matrix(kx, ky, kz, config))
    eigenvalue = _select_low_beta_eigenvalue(eigenvalues, mode)
    return state, backend, grid, fft, workspace, mask, eigenvalue


def _advance_low_beta(
    state: State,
    *,
    steps: int,
    dt: float,
    config: Config,
    grid: object,
    fft: FFTManager,
    workspace: Workspace,
    mask: object,
) -> State:
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    current = state
    for _ in range(steps):
        current = ssprk3_step(current, dt, low_beta_stratified.ideal_rhs, rhs_kwargs=rhs_kwargs)
    return current


def _scaled_state(state: State, factor: complex) -> State:
    out = state.copy()
    for name in out.field_names:
        out[name][...] *= factor
    return out


def _write_low_beta_input(
    path: Path,
    *,
    tmax: float = 0.01,
    dt: float = 0.001,
    t_out_scal: float = 0.001,
    t_out_spec: float = 0.0,
    t_out_full: float = 0.0,
) -> None:
    path.write_text(
        f"""
title = "Low-beta stratified test"
output_dir = "outputs"

[equations]
type = "low_beta_stratified"

[grid]
Nx = 8
Ny = 8
Nz = 8

[time]
tmax = {tmax}
dt_init = {dt}
dt_max = {dt}
use_variable_dt = false

[output]
t_out_scal = {t_out_scal}
t_out_spec = {t_out_spec}
t_out_full = {t_out_full}

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
amplitude = 0.05
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _read_scalar_rows(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


def test_equation_set_selection_s09() -> None:
    assert "s09" in available_equation_sets()
    assert get_equation_module("s09") is s09
    assert Config(equation_set="s09").field_names == s09.FIELD_NAMES


def test_equation_set_selection_low_beta_stratified() -> None:
    module = get_equation_module("low_beta_stratified")
    assert module is low_beta_stratified
    assert Config(equation_set="low_beta_stratified").field_names == ["psi", "omega", "a"]


def test_unknown_equation_set_gives_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown equation set 'not_real'"):
        get_equation_module("not_real")


def test_dissipation_parsing_matches_selected_field_names(tmp_path) -> None:
    input_file = tmp_path / "low_beta.input"
    input_file.write_text(
        """
[equations]
type = "low_beta_stratified"

[grid]
Nx = 8
Ny = 8
Nz = 8

[dissipation.a]
nu_perp = 0.01
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = resolve_run_settings(runfile_path=input_file)

    assert settings.config.field_names == ["psi", "omega", "a"]
    assert settings.initial_condition.type == "low_beta_stratified_mode"
    assert settings.config.dissipation["a"]["nu_perp"] == 0.01
    assert "upar" not in settings.config.dissipation

    bad_file = tmp_path / "bad_low_beta.input"
    bad_file.write_text(
        """
[equations]
type = "low_beta_stratified"

[grid]
Nx = 8
Ny = 8
Nz = 8

[dissipation.upar]
nu_perp = 0.01
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected keys"):
        resolve_run_settings(runfile_path=bad_file)


@pytest.mark.parametrize(
    "kx,ky,kz,N2,vA",
    [
        (1.0, 1.0, 2.0, 0.25, 1.3),
        (0.0, 1.0, 0.0, 0.25, 1.3),
    ],
)
def test_low_beta_linear_matrix_eigenvalues_match_dispersion_relation(
    kx: float,
    ky: float,
    kz: float,
    N2: float,
    vA: float,
) -> None:
    config = Config(equation_set="low_beta_stratified", N2=N2, vA=vA)
    matrix = low_beta_stratified.linear_matrix(kx, ky, kz, config)
    eigenvalues = np.linalg.eigvals(matrix)
    kperp2 = kx**2 + ky**2
    lambda2 = N2 * ky**2 / kperp2 - vA**2 * kz**2
    expected_pair = (
        np.sqrt(lambda2) if lambda2 >= 0.0 else 1j * np.sqrt(-lambda2),
        -np.sqrt(lambda2) if lambda2 >= 0.0 else -1j * np.sqrt(-lambda2),
    )

    assert np.min(np.abs(eigenvalues - expected_pair[0])) < 1.0e-12
    assert np.min(np.abs(eigenvalues - expected_pair[1])) < 1.0e-12
    assert np.min(np.abs(eigenvalues)) < 1.0e-12


def test_low_beta_single_mode_matches_linear_evolution_stable_case() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.3, N2=0.25)
    state0, backend, grid, fft, workspace, mask, eigenvalue = _low_beta_mode_state(
        config,
        k_indices=(1, 1, 2),
        mode="stable_plus",
    )
    dt = 1.0e-3
    steps = 20
    evolved = _advance_low_beta(
        state0,
        steps=steps,
        dt=dt,
        config=config,
        grid=grid,
        fft=fft,
        workspace=workspace,
        mask=mask,
    )
    exact = _scaled_state(state0, np.exp(eigenvalue * steps * dt))

    for name in state0.field_names:
        np.testing.assert_allclose(backend.to_numpy(evolved[name]), backend.to_numpy(exact[name]), rtol=1.0e-6, atol=1.0e-8)


def test_low_beta_single_mode_matches_linear_evolution_unstable_case() -> None:
    config = Config(equation_set="low_beta_stratified", Nx=8, Ny=8, Nz=8, backend="numpy", vA=1.0, N2=0.25)
    state0, backend, grid, fft, workspace, mask, eigenvalue = _low_beta_mode_state(
        config,
        k_indices=(0, 1, 0),
        mode="unstable_growing",
    )
    dt = 1.0e-3
    steps = 50
    evolved = _advance_low_beta(
        state0,
        steps=steps,
        dt=dt,
        config=config,
        grid=grid,
        fft=fft,
        workspace=workspace,
        mask=mask,
    )
    exact = _scaled_state(state0, np.exp(eigenvalue * steps * dt))

    for name in state0.field_names:
        np.testing.assert_allclose(backend.to_numpy(evolved[name]), backend.to_numpy(exact[name]), rtol=1.0e-6, atol=1.0e-8)


def test_low_beta_ideal_budget_source_term_present(tmp_path) -> None:
    input_file = tmp_path / "low_beta_budget.input"
    _write_low_beta_input(input_file)

    main([str(input_file)])

    fieldnames, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    assert "total_energy_rhs_stratification" in fieldnames
    stratification = np.asarray([row["total_energy_rhs_stratification"] for row in rows[1:]], dtype=np.float64)
    assert np.all(np.isfinite(stratification))
    assert np.any(np.abs(stratification) > 1.0e-12)


def test_low_beta_energy_budget_matches_finite_difference(tmp_path) -> None:
    input_file = tmp_path / "low_beta_budget_fd.input"
    _write_low_beta_input(input_file, tmax=0.01, dt=0.001, t_out_scal=0.001)

    main([str(input_file)])

    _, rows = _read_scalar_rows(tmp_path / "outputs" / "scalar_diagnostics.csv")
    time = np.asarray([row["time"] for row in rows], dtype=np.float64)
    energy = np.asarray([row["total_energy"] for row in rows], dtype=np.float64)
    rhs_total = np.asarray([row["total_energy_rhs_total"] for row in rows], dtype=np.float64)

    measured = np.diff(energy) / np.diff(time)
    assert np.allclose(measured, rhs_total[1:], rtol=0.05, atol=1.0e-8)


def test_low_beta_outputs_and_budget_vis_smoke(tmp_path) -> None:
    input_file = tmp_path / "low_beta_outputs.input"
    _write_low_beta_input(input_file, tmax=0.005, dt=0.001, t_out_scal=0.001, t_out_spec=0.005)

    main([str(input_file)])

    output_dir = tmp_path / "outputs"
    scalar_path = output_dir / "scalar_diagnostics.csv"
    spectra_path = output_dir / "spectra.csv"
    assert scalar_path.exists()
    assert spectra_path.exists()

    with spectra_path.open("r", encoding="utf-8", newline="") as handle:
        quantities = {row["quantity"] for row in csv.DictReader(handle)}
    assert {"u_perp", "b_perp", "a"}.issubset(quantities)

    figure_path = tmp_path / "low_beta_budget.png"
    assert plot_budget_main([str(scalar_path), "--output", str(figure_path)]) == figure_path.resolve()
    assert figure_path.exists()


def test_cli_only_low_beta_mode_runs(tmp_path) -> None:
    output_dir = tmp_path / "cli_outputs"

    main(
        [
            "--equation-set",
            "low_beta_stratified",
            "--output-dir",
            str(output_dir),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--tmax",
            "0.001",
            "--dt-init",
            "0.001",
            "--no-use-variable-dt",
            "--t-out-scal",
            "0.001",
            "--t-out-spec",
            "0.0",
            "--t-out-full",
            "0.0",
        ]
    )

    fieldnames, rows = _read_scalar_rows(output_dir / "scalar_diagnostics.csv")
    assert rows
    assert "a_rms" in fieldnames
    assert "total_energy_rhs_stratification" in fieldnames


def test_low_beta_fullfield_output_uses_low_beta_fields(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    input_file = tmp_path / "low_beta_full.input"
    _write_low_beta_input(input_file, tmax=0.001, dt=0.001, t_out_scal=0.0, t_out_full=0.001)

    main([str(input_file)])

    snapshot_path = tmp_path / "outputs" / "fullfields" / "fullfield_0001.h5"
    assert snapshot_path.exists()
    with h5py.File(snapshot_path, "r") as handle:
        field_names = {name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in handle["metadata"]["field_names"][()]}
        assert field_names == {"psi", "omega", "a"}
        output = handle["output"]
        assert {"psi", "omega", "a"}.issubset(output.keys())
