from __future__ import annotations

from pathlib import Path

from rmhdgpu.run import main


def _write_small_input_file(path: Path, *, output_dir_line: str | None = None) -> None:
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
    input_file = case_dir / "input.input"
    _write_small_input_file(input_file, output_dir_line='output_dir = "outputs"')

    main([str(input_file)])

    output_dir = case_dir / "outputs"
    assert output_dir.exists()
    assert (output_dir / "resolved_config.toml").exists()
    assert (output_dir / "run.log").exists()
    assert (output_dir / "scalar_diagnostics.csv").exists()
    assert (output_dir / "input_copy.input").exists()


def test_output_dir_defaults_relative_to_input_file(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    other_dir = tmp_path / "other"
    case_dir.mkdir()
    other_dir.mkdir()
    input_file = case_dir / "input.input"
    _write_small_input_file(input_file)

    monkeypatch.chdir(other_dir)
    main([str(input_file)])

    assert (case_dir / "outputs").exists()
    assert not (other_dir / "outputs" / "resolved_config.toml").exists()


def test_example_runfile_executes_small_case(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "forced_turbulence.input"
    )
    case_dir = tmp_path / "example_case"
    case_dir.mkdir()
    input_file = case_dir / "input.input"
    input_file.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    main([str(input_file), "--tmax", "0.02", "--nx", "8", "--ny", "8", "--nz", "8"])

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
