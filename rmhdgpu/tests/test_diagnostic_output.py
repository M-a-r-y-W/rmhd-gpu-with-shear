from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rmhdgpu.run import main


def _write_input_file(
    path: Path,
    *,
    t_out_scal: float = 0.01,
    t_out_spec: float = 0.0,
    t_out_full: float = 0.0,
    tmax: float = 0.02,
    dt: float = 0.005,
) -> None:
    path.write_text(
        f"""
title = "Diagnostics test"
output_dir = "outputs"

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
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_scalar_csv_written(tmp_path) -> None:
    input_file = tmp_path / "scalar.input"
    _write_input_file(input_file, t_out_scal=0.01, t_out_spec=0.0, t_out_full=0.0)

    main([str(input_file)])

    csv_path = tmp_path / "outputs" / "scalar_diagnostics.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "time" in reader.fieldnames
    assert "step" in reader.fieldnames
    assert "alfvenic_energy" in reader.fieldnames
    assert rows


def test_spectra_csv_written(tmp_path) -> None:
    input_file = tmp_path / "spectra.input"
    _write_input_file(input_file, t_out_scal=0.0, t_out_spec=0.01, t_out_full=0.0)

    main([str(input_file)])

    csv_path = tmp_path / "outputs" / "spectra.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["time", "step", "quantity", "kperp", "value"]
    assert rows
    assert {row["quantity"] for row in rows}


def test_output_cadences_respected(tmp_path) -> None:
    input_file = tmp_path / "cadence.input"
    _write_input_file(input_file, t_out_scal=0.01, t_out_spec=0.015, t_out_full=0.0, tmax=0.03, dt=0.005)

    main([str(input_file)])

    scalar_path = tmp_path / "outputs" / "scalar_diagnostics.csv"
    spectra_path = tmp_path / "outputs" / "spectra.csv"

    with scalar_path.open("r", encoding="utf-8", newline="") as handle:
        scalar_rows = list(csv.DictReader(handle))
    with spectra_path.open("r", encoding="utf-8", newline="") as handle:
        spectra_rows = list(csv.DictReader(handle))

    scalar_times = [float(row["time"]) for row in scalar_rows]
    spectra_times = sorted({float(row["time"]) for row in spectra_rows})

    assert scalar_times == [0.0, 0.01, 0.02, 0.03]
    assert spectra_times == [0.0, 0.015, 0.03]


def test_fullfield_hdf5_written(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    input_file = tmp_path / "full.input"
    _write_input_file(input_file, t_out_scal=0.0, t_out_spec=0.0, t_out_full=0.01)

    main([str(input_file)])

    fullfields_dir = tmp_path / "outputs" / "fullfields"
    snapshot_path = fullfields_dir / "fullfield_0001.h5"
    assert fullfields_dir.exists()
    assert snapshot_path.exists()
    with h5py.File(snapshot_path, "r") as handle:
        assert "metadata" in handle
        assert "output" in handle
        first = handle["output"]
        assert "time" in first
        assert "step" in first
        assert "psi" in first
        assert "omega" in first
