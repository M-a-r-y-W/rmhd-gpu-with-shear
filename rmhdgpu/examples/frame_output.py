"""Small helpers for writing signed x-y frame sequences from example runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rmhdgpu.operators import lap_perp


DEFAULT_FRAME_COUNT = 12


def positive_int(value: str) -> int:
    """Parse a strictly positive integer argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    """Add conservative movie-frame options to an example parser."""

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Write x-y PNG frames of vorticity and current for later movie assembly.",
    )
    parser.add_argument(
        "--frame-count",
        type=positive_int,
        default=DEFAULT_FRAME_COUNT,
        help="Number of evenly spaced frame times between t=0 and t_final.",
    )
    parser.add_argument(
        "--snapshot-z-index",
        type=int,
        default=None,
        help="z-plane index for x-y snapshots. Defaults to the midplane.",
    )


def resolve_snapshot_z_index(grid: Any, requested: int | None) -> int:
    """Return a valid z-index for x-y snapshots."""

    if requested is None:
        return grid.Nz // 2
    if requested < 0 or requested >= grid.Nz:
        raise ValueError(f"snapshot z-index {requested} is out of range for Nz={grid.Nz}")
    return requested


def build_frame_times(t_final: float, frame_count: int) -> np.ndarray:
    """Return evenly spaced frame times including the endpoints."""

    return np.linspace(0.0, float(t_final), frame_count, dtype=np.float64)


def capture_xy_signed_fields(
    state: Any,
    *,
    time: float,
    grid: Any,
    fft: Any,
    backend: Any,
    z_index: int,
) -> dict[str, Any]:
    """Extract NumPy x-y slices of vorticity and current at one time."""

    omega_xy = backend.to_numpy(fft.c2r(state["omega"]))[:, :, z_index].copy()
    current_xy = backend.to_numpy(fft.c2r(-lap_perp(state["psi"], grid)))[:, :, z_index].copy()
    return {
        "time": float(time),
        "vorticity": omega_xy,
        "current": current_xy,
    }


def write_signed_xy_frames(
    frames: list[dict[str, Any]],
    *,
    output_dir: Path,
    grid: Any,
    z_index: int,
) -> None:
    """Write vorticity/current PNG sequences with fixed color limits per quantity."""

    if not frames:
        return

    x_min = 0.0
    x_max = float(grid.Lx)
    y_min = 0.0
    y_max = float(grid.Ly)
    z_value = float(grid.dz * z_index)
    quantities = [
        ("vorticity", "Vorticity"),
        ("current", "Current"),
    ]

    for quantity, title in quantities:
        frame_dir = output_dir / f"frames_{quantity}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        vmax = max(float(np.max(np.abs(frame[quantity]))) for frame in frames)
        if vmax <= 0.0:
            vmax = 1.0

        print(f"Writing {quantity} frames to {frame_dir}")
        for index, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            image = ax.imshow(
                frame[quantity].T,
                origin="lower",
                extent=(x_min, x_max, y_min, y_max),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title} at t={frame['time']:.3f}, z={z_value:.3f} (index {z_index})")
            fig.colorbar(image, ax=ax, label=title)
            frame_path = frame_dir / f"{quantity}_{index:04d}.png"
            fig.savefig(frame_path, dpi=160)
            plt.close(fig)

        print(
            "ffmpeg hint:",
            f"ffmpeg -framerate 10 -i '{frame_dir / (quantity + '_%04d.png')}' "
            f"-pix_fmt yuv420p '{frame_dir / (quantity + '.mp4')}'",
        )
