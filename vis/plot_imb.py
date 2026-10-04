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
from vis.plot_energy_shear import rolling_cv


def load_run(run_dir: Path) -> dict:
    """Extract energies for one  from one alpha value from one simulation."""
    run_dir = Path(run_dir)

    with (run_dir / "resolved_config.toml").open("rb") as f:
        config = tomllib.load(f)

    fieldnames, columns = _read_scalar_csv(run_dir / "scalar_diagnostics.csv")

    time = columns["time"] if "time" in columns else columns["t"]

    idx= rolling_cv(columns["elsasser_energy_ratio"],50, 0.1, 10)
    if idx == None:
        print(f"No steady state in energy ratio detected for {run_dir.name}")
    Av_values= columns["elsasser_energy_ratio"][idx:]
    Avg= float(np.mean(Av_values))
    std= float(np.std(Av_values))
    plus= config["forcing"]["epsilon_plus"]
    minus= config["forcing"]["epsilon_minus"] 
    alpha = plus/minus
    
    idex= rolling_cv(columns["normalized_cross_helicity"],50, 0.1, 10)
    if idex == None:
        print(f"No steady state in energy ratio detected for {run_dir.name}")
    Avg_values= columns["normalized_cross_helicity"][idex:]
    Avg_helicity= float(np.mean(Avg_values))
    Ratio= 1+Avg_helicity/(1-Avg_helicity)
    
    return {
        "run": run_dir.name,
        "alpha": alpha,
        "ratio": Avg,
        "helicityratio": Ratio,
        "std": std
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
    helicityratio= np.array([r["helicityratio"] for r in rows])
    std= np.array([r["std"] for r in rows])

    theory_alpha= np.linspace(0, max(alpha), 200)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    
    ax.errorbar(alpha, ratio, yerr=std, fmt='o', label= "Numerical")
    ax.errorbar(alpha, helicityratio, yerr=std, fmt='o', label= "Numerical")
    ax.plot(theory_alpha, theory_alpha**2, lw=3, ls= "--", color= "black", label="Analytical")
    
    ax.set_xlabel(r"$\alpha_\epsilon$", fontsize=18)
    ax.set_ylabel(r"$\alpha$", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    #ax.set_title("Ratio of upar energy to dbpar energy: numerical vs theoretical")
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    output_path = (
            Path.cwd() / "Imbalanced_comp.png" if args.output is None
            else Path(args.output).expanduser().resolve()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True) # need more fiddling with output path if have time so stop sending it to rmhdgpu-with-shear directory

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()

