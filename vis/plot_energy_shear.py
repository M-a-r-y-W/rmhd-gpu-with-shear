"""Plots the rhs_shear energy rate against the dissipation rate and determines the steady state dissipation rate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis._matplotlib import finalize_figure, import_pyplot


def _read_scalar_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Scalar diagnostics file {path} has no header row.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Scalar diagnostics file {path} contains no data rows.")

    columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
    for row in rows:
        for name in reader.fieldnames:
            columns[name].append(float(row[name]))
    return list(reader.fieldnames), {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to scalar_diagnostics.csv.")
    parser.add_argument(
        "--quantity",
        default="total_energy",
        help="Conserved quantity name prefix, for example `total_energy`.",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults next to the CSV file.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving. Useful from Spyder or IPython.",
    )
    return parser

def rolling_cv(x:np.array, window:int, cv_threshold:float, consecutive:int)-> int|None:
    """ Computes the rolling coefficient of variation (CV) of the energies over a set window of timesteps. 
    Once the CV drops under a set threshold for a set number of consecutive windows, the function returns the first index in the consecutive range where the CV dropped below the threshold.
    If the CV never drops below the threshold, the output is None.
    """
    cv= np.full(len(x), np.nan) # computing indexes that don't exist leaves NaN so fine
    for i in range(window,len(x)):
        window_slice= x[i-window:i]
        mean= float(np.mean(window_slice))
        std= float(np.std(window_slice))
        if mean != 0:
          cv[i]= abs(std/mean)
        else: cv[i]= np.inf
    below =cv < cv_threshold
    for i in range(window,len(x)-consecutive):
        if np.all(below[i:i+consecutive]):
           return i
    return None


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    csv_path = Path(args.csv_path).expanduser().resolve()
    fieldnames, columns = _read_scalar_csv(csv_path)
    time_key = "time" if "time" in columns else "t"
    time = columns[time_key]

    if args.quantity not in columns:
        raise SystemExit(f"Quantity {args.quantity!r} is not present in {csv_path}.")

    rhs_term_names = sorted(
        name
        for name in fieldnames
        if name == f"{args.quantity}_rhs_dissipation" or name == f"{args.quantity}_rhs_shear"
    )

    steady_state_rate= np.empty(len(rhs_term_names))
    for Index, E_names in enumerate(rhs_term_names):
        idx= rolling_cv(columns[E_names], 30, 0.05, 10)
        if idx == None:
           raise ValueError(f"No steady state detected for {E_names}")
        Av_values= columns[E_names][idx:]
        steady_state_rate[Index]= np.mean(Av_values) 

    
    output_path = (
        csv_path.with_name(f"{args.quantity}_budget.png")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots()
  
    term_linestyles = ["--", ":"]
    for index, term_name in enumerate(rhs_term_names):
        axes.plot(
            time,
            columns[term_name],
            lw=1.8,
            ls=term_linestyles[index % len(term_linestyles)],
            label=term_name,
        )
    for idex, term_names in enumerate(rhs_term_names):
        axes.axhline(steady_state_rate[idex],label= f"Steady state rate for {term_names} ={steady_state_rate[idex]:.3f}")
        
    axes.axhline(0.0, color="0.4", lw=1.0, alpha=0.6)
    axes.set_xlabel("time")
    axes.set_ylabel(r"shear and dissipation terms / d$_t Q$")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()
