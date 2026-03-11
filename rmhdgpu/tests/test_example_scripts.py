from __future__ import annotations

import sys
from types import SimpleNamespace

from rmhdgpu.examples.frame_output import build_frame_times, resolve_snapshot_z_index
from rmhdgpu.examples.sanity_aw_packet import build_parser as build_aw_parser
from rmhdgpu.examples.sanity_aw_packet import main as aw_main
from rmhdgpu.examples.sanity_decay_spectra import (
    GPU_256_T_FINAL as DECAY_GPU_256_T_FINAL,
)
from rmhdgpu.examples.sanity_decay_spectra import (
    build_parser as build_decay_parser,
)
from rmhdgpu.examples.sanity_decay_spectra import (
    main as decay_main,
)
from rmhdgpu.examples.sanity_decay_spectra import (
    resolve_run_parameters as resolve_decay_run_parameters,
)
from rmhdgpu.examples.sanity_forced_turbulence import (
    GPU_256_T_FINAL as FORCED_GPU_256_T_FINAL,
)
from rmhdgpu.examples.sanity_forced_turbulence import (
    build_parser as build_forced_parser,
)
from rmhdgpu.examples.sanity_forced_turbulence import (
    main as forced_main,
)
from rmhdgpu.examples.sanity_forced_turbulence import (
    resolve_run_parameters as resolve_forced_run_parameters,
)


def test_decay_gpu_256_preset_uses_cupy_large_grid_and_short_runtime() -> None:
    args = build_decay_parser().parse_args(["--gpu-256"])
    params = resolve_decay_run_parameters(args)

    assert params["backend_name"] == "cupy"
    assert params["n"] == 256
    assert params["fft_workers"] is None
    assert params["t_final"] == DECAY_GPU_256_T_FINAL


def test_forced_gpu_256_preset_uses_cupy_large_grid_and_short_runtime() -> None:
    args = build_forced_parser().parse_args(["--gpu-256"])
    params = resolve_forced_run_parameters(args)

    assert params["backend_name"] == "cupy"
    assert params["n"] == 256
    assert params["fft_workers"] is None
    assert params["t_final"] == FORCED_GPU_256_T_FINAL


def test_aw_parser_accepts_frame_arguments() -> None:
    args = build_aw_parser().parse_args(["--save-frames", "--frame-count", "5", "--snapshot-z-index", "3"])

    assert args.save_frames is True
    assert args.frame_count == 5
    assert args.snapshot_z_index == 3


def test_frame_time_builder_and_z_index_validation() -> None:
    frame_times = build_frame_times(1.5, 4)

    assert frame_times.tolist() == [0.0, 0.5, 1.0, 1.5]
    assert resolve_snapshot_z_index(SimpleNamespace(Nz=8), None) == 4
    assert resolve_snapshot_z_index(SimpleNamespace(Nz=8), 2) == 2


def test_aw_packet_smoke_run_writes_summary_and_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_aw_packet",
            "--output-dir",
            str(tmp_path),
            "--nx",
            "8",
            "--save-frames",
            "--frame-count",
            "2",
            "--snapshot-z-index",
            "0",
        ],
    )

    aw_main()

    assert (tmp_path / "sanity_aw_packet.png").exists()
    assert (tmp_path / "frames_vorticity" / "vorticity_0000.png").exists()
    assert (tmp_path / "frames_current" / "current_0001.png").exists()


def test_decay_spectra_smoke_run_writes_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_decay_spectra",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "numpy",
            "--n",
            "8",
            "--t-final",
            "0.05",
        ],
    )

    decay_main()

    assert (tmp_path / "sanity_decay_spectra.png").exists()


def test_forced_turbulence_smoke_run_writes_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sanity_forced_turbulence",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "numpy",
            "--n",
            "8",
            "--t-final",
            "0.05",
            "--forcing-seed",
            "7",
        ],
    )

    forced_main()

    assert (tmp_path / "sanity_forced_turbulence.png").exists()
