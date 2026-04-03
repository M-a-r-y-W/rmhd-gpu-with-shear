from __future__ import annotations

import pytest

from rmhdgpu.run import build_parser
from rmhdgpu.runfile import cli_overrides_from_args, load_run_file, resolve_run_settings


def test_minimal_runfile_parses(tmp_path) -> None:
    runfile = tmp_path / "minimal.run"
    runfile.write_text('title = "Minimal case"\n', encoding="utf-8")

    settings = resolve_run_settings(runfile_path=runfile)

    assert settings.title == "Minimal case"
    assert settings.config.Nx == 16
    assert settings.config.backend == "numpy"
    assert settings.output_dir == (tmp_path / "outputs").resolve()
    assert settings.initial_condition.type == "alfven_mode"


def test_nested_sections_parse_correctly(tmp_path) -> None:
    runfile = tmp_path / "nested.run"
    runfile.write_text(
        """
title = "Nested case"
output_dir = "case_outputs"

[grid]
Nx = 32
Ny = 16
Nz = 8

[time]
tmax = 2.5
dt_init = 0.01
cfl_number = 0.2

[backend]
backend = "scipy_cpu"
fft_workers = 4

[forcing]
use_forcing = true
forcing_seed = 7

[forcing.force_amplitudes]
psi = 0.05

[dissipation.psi]
nu_perp = 0.005
nu_par = 0.0
n_perp = 2
n_par = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=runfile)

    assert settings.config.Nx == 32
    assert settings.config.Ny == 16
    assert settings.config.Nz == 8
    assert settings.config.tmax == 2.5
    assert settings.config.backend == "scipy_cpu"
    assert settings.config.fft_workers == 4
    assert settings.config.use_forcing is True
    assert settings.config.forcing_seed == 7
    assert settings.config.force_amplitudes["psi"] == 0.05
    assert settings.config.force_amplitudes["omega"] == 0.0
    assert settings.config.dissipation["psi"]["nu_perp"] == 0.005
    assert settings.config.dissipation["omega"]["nu_perp"] == 0.0


def test_invalid_runfile_gives_helpful_error(tmp_path) -> None:
    runfile = tmp_path / "bad.run"
    runfile.write_text("[grid]\nNx = [1, 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse TOML run file"):
        load_run_file(runfile)


def test_cli_overrides_runfile(tmp_path) -> None:
    runfile = tmp_path / "override.run"
    runfile.write_text(
        """
[grid]
Nx = 16
Ny = 16
Nz = 16

[time]
tmax = 1.0

[backend]
backend = "numpy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args([str(runfile), "--backend", "cupy", "--tmax", "10.0"])
    settings = resolve_run_settings(
        runfile_path=args.runfile,
        cli_overrides=cli_overrides_from_args(args),
    )

    assert settings.config.backend == "cupy"
    assert settings.config.tmax == 10.0


def test_runfile_without_cli_matches_expected_config(tmp_path) -> None:
    runfile = tmp_path / "forcing.run"
    runfile.write_text(
        """
[grid]
Nx = 8
Ny = 8
Nz = 8

[forcing]
use_forcing = true
forcing_seed = 22

[forcing.force_amplitudes]
psi = 0.02
omega = 0.03

[initial_condition]
type = "zero"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings = resolve_run_settings(runfile_path=runfile)

    assert settings.config.use_forcing is True
    assert settings.config.forcing_seed == 22
    assert settings.config.force_amplitudes["psi"] == 0.02
    assert settings.config.force_amplitudes["omega"] == 0.03
    assert settings.initial_condition.type == "zero"
