""" Plot the ratio of the dbpar energy to the upar energy (unweighted) and compared to expected scaling of slaved estimate. """
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
from vis.plot_scalars import _read_scalar_csv  


def load_run(run_dir: Path) -> dict:
    """Extract energies for one  from one alpha value from one simulation."""
    run_dir = Path(run_dir)

    with (run_dir / "resolved_config.toml").open("rb") as f:
        config = tomllib.load(f)

    fieldnames, columns = _read_scalar_csv(run_dir / "scalar_diagnostics.csv")

    time = columns["time"] if "time" in columns else columns["t"]
    dbpar_energy = columns["dbpar_energy"]

    return {
        "run": run_dir.name,
        "time": time,
        "dbpar_energy": dbpar_energy,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Path to simulation output directories, one per simulation.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults next to the CSV file.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving. Useful from Spyder or IPython.",
    )
    return parser

def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)

    rows=[load_run(Path(d)) for d in args.run_dirs]
    rows.sort(key=lambda r: r["run"]) # orders in terms of alpha value


    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    chi_A= [r"$\chi_A= 8.1$", r"$\chi_A= 4.9$", r"$\chi_A= 1.6$", r"$\chi_A= 0.85$", r"$\chi_A= 0.26$"]
    for i, row in enumerate(rows):
            ax.plot(row["time"], row["dbpar_energy"], lw=2, label=chi_A[i])
    
    ax.set_xlabel(r"Time / $\tau_A$", fontsize=18)
    ax.set_ylabel("Energy Density", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    output_path = (
            Path.cwd() / "dbpar_comparison.png" if args.output is None
            else Path(args.output).expanduser().resolve()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True) # need more fiddling with output path if have time so stop sending it to rmhdgpu-with-shear directory

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()

