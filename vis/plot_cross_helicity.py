"""Plot scalar diagnostics of the cross helicity saved by `rmhdgpu.run`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vis.plot_energy_shear import rolling_cv
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


def _default_columns(fieldnames: list[str]) -> list[str]:
    preferred = [
        "normalized_cross_helicity",
        "elsasser_energy_ratio",
    ]
    selected = [name for name in preferred if name in fieldnames]
    if selected:
        return selected
    return [name for name in fieldnames if name not in {"time", "t", "step", "dt"}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to scalar_diagnostics.csv.")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to plot. Defaults to common energy-like diagnostics when present.",
    )
    parser.add_argument("--output", default=None, help="Output image path. Defaults next to the CSV file.")
    parser.add_argument("--title", default="Scalar diagnostics", help="Figure title.")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a log y scale. Only columns that stay strictly positive are plotted in this mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving. Useful from Spyder or IPython.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    plt = import_pyplot(show=args.show)
    csv_path = Path(args.csv_path).expanduser().resolve()
    fieldnames, columns = _read_scalar_csv(csv_path)
    time_key = "time" if "time" in columns else "t"
    time = columns[time_key]

    selected_columns = _default_columns(fieldnames) if args.columns is None else list(args.columns)
    if not selected_columns:
        raise SystemExit("No scalar columns were selected for plotting.")

    unknown = [name for name in selected_columns if name not in columns]
    if unknown:
        raise SystemExit(f"Unknown scalar columns requested: {unknown}.")

    output_path = (
        csv_path.with_name("scalar_diagnostics_cross_helicity.png")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotted_columns = selected_columns
    if args.log:
        plotted_columns = [name for name in selected_columns if np.all(columns[name] > 0.0)]
        if not plotted_columns:
            raise SystemExit("No strictly positive scalar columns were selected for --log plotting.")

    steady_state_rate= np.full(len(plotted_columns), np.nan)
    for Index, names in enumerate(plotted_columns):
        idx= rolling_cv(columns[names], 50, 0.075, 10)
        if idx == None:
            print(f"No steady state detected for {names}")
            continue
        Av_values= columns[names][idx:]
        steady_state_rate[Index]= np.mean(Av_values) 

    fig, axes = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colours = ["tab:red", "tab:blue"]

    for index, term_name in enumerate(plotted_columns):
            axes.plot(
                time,
                columns[term_name],
                lw=3,
                ls="-",
                label=term_name,
                color=colours[index % len(colours)],
            )
    label=[r"$\sigma_c$", r"$E_{+}/E_{-}$"]
    for idex, term_names in enumerate(plotted_columns):
        if not np.isnan(steady_state_rate[idex]):
         axes.axhline(steady_state_rate[idex],label= label[idex]+ f"= {steady_state_rate[idex]:.3f}", color=colours[idex % len(colours)], lw=3, ls="--")

    axes.set_xlabel(r"Time / $\tau_A$", fontsize=18)
    axes.set_ylabel("Energy Density", fontsize=18)
    axes.tick_params(axis="both", labelsize=14)
    #axes.set_title("Scalar Diagnostics")
    #if args.log:
    #axes.set_yscale("log")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=14)

    finalize_figure(fig, output_path=output_path, show=args.show, plt=plt)
    return output_path


if __name__ == "__main__":
    main()
