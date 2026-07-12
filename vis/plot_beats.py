"""Plot linear slow modes from 1D slices against the theoretical solution for slow modes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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
        raise SystemExit("plot_linear.py requires `h5py` to read full-field HDF5 snapshots.")


def _resolve_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("fullfield_*.h5"))
        if not files:
            raise SystemExit(f"No full-field snapshot files were found in {path}.")
        return files
    if path.suffix != ".h5":
        raise SystemExit(f"Expected a snapshot .h5 file or a directory of snapshots; got {path}.")
    return [path]

def _extract_point(field: np.ndarray, *, x_index: int, y_index: int, z_index:int) -> np.ndarray:
    return field[x_index, y_index, z_index]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Path to a full-field snapshot `.h5` file or to a directory of `fullfield_*.h5` files.",
    )
    parser.add_argument("--field1", default="upar", help="Field to plot.")
    parser.add_argument("--field2", default="dbpar", help="Field to plot.")
    parser.add_argument("--x-index", type=int, default=None, help="Explicit x index.")
    parser.add_argument("--y-index", type=int, default=None, help="Explicit y index.")
    parser.add_argument("--z-index", type=int, default=None, help="Explicit z index.")

    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of snapshot numbers to plot, matching file names such as `0001`.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs.")

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Useful from Spyder or IPython.",
    )
    return parser

def _read_resolved_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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

    config = _read_resolved_config(
        input_path.parent / "resolved_config.toml"
        if input_path.is_dir()
        else input_path.parent.parent / "resolved_config.toml"
    )
    saved_paths: list[Path] = []

    physics = config["physics"]
    vA = physics["vA"]
    chi = physics["cs2_over_vA2"]
    Ku = physics["Ku"]
    alpha = chi / (1.0 + chi)

    vslow = vA * np.sqrt(alpha)

    grid= config["grid"]
    Nx = grid["Nx"]
    Ny = grid["Ny"] 
    Nz = grid["Nz"]
    Lx = grid["Lx"]
    Ly = grid["Ly"] 
    Lz = grid["Lz"]
    
    initial_condition_type = config["initial_condition"]["type"]
    if initial_condition_type != "alfven_mode":
        raise SystemExit(
            "This plotting script requires initial_condition.type = 'alfven_mode'. "
            f"Found '{initial_condition_type}' instead."
        )
    
    if "parameters" not in config["initial_condition"]: 
        raise SystemExit(
            "This plotting script requires initial_condition.parameters to be specified."
        )
    ic = config["initial_condition"]["parameters"]
    kx_mode,ky_mode,kz_mode = ic["k_indices"]
    amp=ic["amplitude"]
    branch = ic["branch"]

    if branch != "minus": 
        raise SystemExit(
            "This plotting script requires branch = 'minus'. "
            f"Found '{branch}' instead."
        )
    
    k_x =2 * np.pi * kx_mode / Lx
    k_y =2 * np.pi * ky_mode / Ly
    k_z =2 * np.pi * kz_mode / Lz
    omega = k_z * vA
    omega_slow =k_z * vslow
    Ku_scaled= 0.5 *Ku *(1.0- vslow/vA)
    kperp= np.sqrt(k_x ** 2 + k_y ** 2)
    zperp= 1j* 2.0 *amp *k_y *np.sqrt(2.0)/kperp
    if np.isclose(omega_slow,omega):
        theoretical_amp_r= -zperp* Ku_scaled
    else: theoretical_amp= 1j* zperp* Ku_scaled/(omega_slow-omega)
    
    time=[]
    num_sol=[]
    theo_sol=[]
    for snapshot_path in snapshot_files:
        with h5py.File(snapshot_path, "r") as handle:
            metadata = handle["metadata"]
            output_group = handle["output"]

            x = np.asarray(metadata["x"])
            y = np.asarray(metadata["y"])
            z = np.asarray(metadata["z"])

            if args.field1 not in output_group or args.field2 not in output_group:
                raise SystemExit(f"One or both fields {args.field1!r} and {args.field2!r} are not present in {snapshot_path}.")
            
            time_value = float(np.asarray(output_group["time"]))
            step_value = int(np.asarray(output_group["step"]))
            field1 = np.asarray(output_group[args.field1])
            field2 = np.asarray(output_group[args.field2])

            if args.x_index is None:
                x_index = len(x) // 2
            else: 
                x_index = args.x_index

            if args.y_index is None:
                y_index = len(y) // 2
            else: 
                y_index = args.y_index

            if args.z_index is None:
                z_index = len(z) // 2
            else:
                z_index = args.z_index

            if not (0 <= x_index < len(x)):
                raise SystemExit(f"x_index {x_index} is out of bounds for x array of length {len(x)}.")
            if not (0 <= y_index < len(y)):
                raise SystemExit(f"y_index {y_index} is out of bounds for y array of length {len(y)}.")
            if not (0 <= z_index < len(z)):
                raise SystemExit(f"z_index {z_index} is out of bounds for z array of length {len(z)}.")

            field1_point=_extract_point(field1,x_index=x_index, y_index=y_index, z_index=z_index)
            field2_point=_extract_point(field2,x_index=x_index, y_index=y_index, z_index=z_index)
        
            
            numerical_sol= field1_point - field2_point* vA *1/np.sqrt(alpha)

            x_0= x[x_index]
            y_0=y[y_index]
            z_0= z[z_index]
            t= time_value

            phase_shift= k_x *x_0 + k_y *y_0 + k_z *z_0
            phase_slow= phase_shift-omega_slow*t
            phase_alfven= phase_shift-omega*t

            if np.isclose(omega_slow, omega):
               theoretical_sol=np.real(theoretical_amp_r*time_value*np.exp(1j*phase_alfven))
            else: theoretical_sol= np.real(theoretical_amp * (np.exp(1j*phase_alfven)-np.exp(1j*phase_slow)))
           
            time.append(t)
            num_sol.append(numerical_sol)
            theo_sol.append(theoretical_sol)

    fig,ax= plt.subplots()

    ax.plot(time, num_sol, label="Numerical Solution", color="black")
    ax.plot(time, theo_sol, label="Theoretical Solution", color="purple")
    ax.set_xlabel("t")
    ax.set_ylabel("Amplitude")
    ax.set_title("Comparison of numerical and theoretical linearised slow waves over time")
    ax.legend()

    output_dir = (
                (input_path if input_path.is_dir() else input_path.parent) / "linearised_slow_wave_comparison"
                if args.output_dir is None
                else Path(args.output_dir).expanduser().resolve()
            )
    output_dir.mkdir(parents=True, exist_ok=True)
            
    output_path = output_dir / f"Linearised_slow_wave_beating.png"
    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    saved_paths.append(output_path)
    
    return saved_paths
            
    
if __name__ == "__main__":
    main()
    


    


