"""Plot 2D slices from per-snapshot full-field HDF5 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot

try:
    import h5py
except ImportError:  # pragma: no cover - exercised by runtime error path
    h5py = None


def _require_h5py() -> None:
    if h5py is None:
        raise SystemExit("plot_fullfield.py requires `h5py` to read full-field HDF5 snapshots.")


def _resolve_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("fullfield_*.h5"))
        if not files:
            raise SystemExit(f"No full-field snapshot files were found in {path}.")
        return files
    if path.suffix != ".h5":
        raise SystemExit(f"Expected a snapshot .h5 file or a directory of snapshots; got {path}.")
    return [path]


def _resolve_slice_index(coords: np.ndarray, *, requested_index: int | None, requested_coord: float | None) -> int:
    if requested_coord is not None:
        return int(np.argmin(np.abs(coords - requested_coord)))
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(coords):
            raise SystemExit(f"slice index {requested_index} is out of range for axis of length {len(coords)}.")
        return requested_index
    return len(coords) // 2


def _extract_slice(field: np.ndarray, *, slice_dir: str, slice_index: int) -> np.ndarray:
    if slice_dir == "x":
        return field[slice_index, :, :]
    if slice_dir == "y":
        return field[:, slice_index, :]
    return field[:, :, slice_index]


def _slice_axes(
    *,
    slice_dir: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[str, str, tuple[float, float, float, float]]:
    if slice_dir == "x":
        return "y", "z", (float(y[0]), float(y[-1]), float(z[0]), float(z[-1]))
    if slice_dir == "y":
        return "x", "z", (float(x[0]), float(x[-1]), float(z[0]), float(z[-1]))
    return "x", "y", (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Path to a full-field snapshot `.h5` file or to a directory of `fullfield_*.h5` files.",
    )
    parser.add_argument("--field", default="psi", help="Field to plot.")
    parser.add_argument("--slice-dir", choices=["x", "y", "z"], default="z", help="Slice direction.")
    parser.add_argument("--slice-index", type=int, default=None, help="Explicit slice index.")
    parser.add_argument(
        "--slice-coordinate",
        type=float,
        default=None,
        help="Slice coordinate. The nearest stored plane is selected.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of snapshot numbers to plot, matching file names such as `0001`.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap name.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    _require_h5py()
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    input_path = Path(args.input_path).expanduser().resolve()
    snapshot_files = _resolve_input_files(input_path)
    if args.indices is not None:
        requested_names = {f"fullfield_{index:04d}.h5" for index in args.indices}
        snapshot_files = [path for path in snapshot_files if path.name in requested_names]
        if not snapshot_files:
            raise SystemExit(f"Requested snapshot numbers were not present in {input_path}.")

    saved_paths: list[Path] = []
    slices: list[tuple[str, float, int, np.ndarray]] = []
    x = y = z = None
    vmax = 0.0
    for snapshot_path in snapshot_files:
        with h5py.File(snapshot_path, "r") as handle:
            metadata = handle["metadata"]
            output_group = handle["output"]

            if x is None:
                x = np.asarray(metadata["x"])
                y = np.asarray(metadata["y"])
                z = np.asarray(metadata["z"])
            if args.field not in output_group:
                raise SystemExit(f"Field {args.field!r} is not present in {snapshot_path}.")

            field = np.asarray(output_group[args.field])
            slice_index = _resolve_slice_index(
                {"x": x, "y": y, "z": z}[args.slice_dir],
                requested_index=args.slice_index,
                requested_coord=args.slice_coordinate,
            )
            slice_data = _extract_slice(field, slice_dir=args.slice_dir, slice_index=slice_index)
            time_value = float(np.asarray(output_group["time"]))
            step_value = int(np.asarray(output_group["step"]))
            vmax = max(vmax, float(np.max(np.abs(slice_data))))
            slices.append((snapshot_path.stem.split("_")[-1], time_value, step_value, slice_data))

    vmax = 1.0 if vmax == 0.0 else vmax
    output_dir = (
        (input_path if input_path.is_dir() else input_path.parent) / f"{args.field}_{args.slice_dir}_slices"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assert x is not None and y is not None and z is not None
    xlabel, ylabel, extent = _slice_axes(slice_dir=args.slice_dir, x=x, y=y, z=z)
    slice_index = _resolve_slice_index(
        {"x": x, "y": y, "z": z}[args.slice_dir],
        requested_index=args.slice_index,
        requested_coord=args.slice_coordinate,
    )

    for key, time_value, step_value, slice_data in slices:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
        image = ax.imshow(
            slice_data.T,
            origin="lower",
            cmap=args.cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(image, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{args.field} {args.slice_dir}={slice_index}, t={time_value:.3f}, step={step_value}")

        output_path = output_dir / f"{args.field}_{args.slice_dir}_{key}.png"
        finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
        saved_paths.append(output_path)

    return saved_paths


if __name__ == "__main__":
    main()
