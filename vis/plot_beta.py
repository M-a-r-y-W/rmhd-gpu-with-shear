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
    upar_energy = columns["upar_energy"]
    dbpar_energy = columns["dbpar_energy"]

    window = 0.1
    time_range= time >= time[-1] - window  
    ratio= float(np.mean(dbpar_energy[time_range]/upar_energy[time_range]))
    std= float(np.std(dbpar_energy[time_range]/upar_energy[time_range]))
    chi= config["physics"]["cs2_over_vA2"]
    alpha = chi / (1.0 + chi)
    
    return {
        "run": run_dir.name,
        "alpha": alpha,
        "ratio": ratio,
        "std":std
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
    rows.sort(key=lambda r: r["alpha"]) # orders in terms of alpha value

    alpha= np.array([r["alpha"] for r in rows])
    ratio= np.array([r["ratio"] for r in rows])
    std= np.array([r["std"] for r in rows])

    theory_alpha= np.linspace(0, max(alpha), 200)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    
    ax.errorbar(alpha, ratio, yerr=std, fmt='o', label= "Numerical")
    ax.plot(theory_alpha, theory_alpha**2, lw=2, ls= "--", color= "black", label="Theoretical")
    
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Unweighted $E_{\delta b_\parallel} / E_{u_\parallel}$")
    ax.set_title("Ratio of upar energy to dbpar energy: numerical vs theoretical")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = (
            Path.cwd() / "Beta_comparison.png" if args.output is None
            else Path(args.output).expanduser().resolve()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()

