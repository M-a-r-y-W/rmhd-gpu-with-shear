from __future__ import annotations

from pathlib import Path

from rmhdgpu.run import main


def _write_small_runfile(path: Path, *, output_dir_line: str | None = None) -> None:
    lines = [
        'title = "Tiny test"',
    ]
    if output_dir_line is not None:
        lines.append(output_dir_line)
    lines.extend(
        [
            "",
            "[grid]",
            "Nx = 8",
            "Ny = 8",
            "Nz = 8",
            "",
            "[time]",
            "tmax = 0.02",
            "dt_init = 0.005",
            "dt_max = 0.01",
            "t_out_scal = 0.01",
            "",
            "[backend]",
            'backend = "numpy"',
            "",
            "[runtime]",
            "progress_output_every = 5",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolved_config_written(tmp_path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    runfile = case_dir / "input.run"
    _write_small_runfile(runfile, output_dir_line='output_dir = "outputs"')

    main([str(runfile)])

    output_dir = case_dir / "outputs"
    assert output_dir.exists()
    assert (output_dir / "resolved_config.toml").exists()
    assert (output_dir / "run.log").exists()
    assert (output_dir / "scalar_diagnostics.csv").exists()
    assert (output_dir / "input_copy.run").exists()


def test_output_dir_defaults_relative_to_input_file(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    other_dir = tmp_path / "other"
    case_dir.mkdir()
    other_dir.mkdir()
    runfile = case_dir / "input.run"
    _write_small_runfile(runfile)

    monkeypatch.chdir(other_dir)
    main([str(runfile)])

    assert (case_dir / "outputs").exists()
    assert not (other_dir / "outputs" / "resolved_config.toml").exists()


def test_example_runfile_executes_small_case(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "input_files"
        / "forced_turbulence_small.run"
    )
    case_dir = tmp_path / "example_case"
    case_dir.mkdir()
    runfile = case_dir / "input.run"
    runfile.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    main([str(runfile), "--tmax", "0.02", "--nx", "8", "--ny", "8", "--nz", "8"])

    output_dir = case_dir / "outputs"
    assert (output_dir / "resolved_config.toml").exists()
    assert (output_dir / "scalar_diagnostics.csv").exists()


def test_cli_only_mode_still_works(tmp_path) -> None:
    output_dir = tmp_path / "cli_outputs"

    main(
        [
            "--backend",
            "numpy",
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--tmax",
            "0.02",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "resolved_config.toml").exists()
    assert (output_dir / "run.log").exists()
    assert (output_dir / "scalar_diagnostics.csv").exists()
