from __future__ import annotations

import csv
from pathlib import Path

from vis.plot_budget import main as plot_budget_main


def test_plot_budget_smoke(tmp_path) -> None:
    csv_path = tmp_path / "scalar_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "step",
                "total_energy",
                "total_energy_rhs_dissipation",
                "total_energy_rhs_forcing",
                "total_energy_rhs_total",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "time": 0.0,
                "step": 0,
                "total_energy": 1.0,
                "total_energy_rhs_dissipation": 0.0,
                "total_energy_rhs_forcing": 0.0,
                "total_energy_rhs_total": 0.0,
            }
        )
        writer.writerow(
            {
                "time": 0.1,
                "step": 1,
                "total_energy": 0.99,
                "total_energy_rhs_dissipation": -0.12,
                "total_energy_rhs_forcing": 0.02,
                "total_energy_rhs_total": -0.10,
            }
        )

    output_path = tmp_path / "budget.png"
    result = plot_budget_main([str(csv_path), "--output", str(output_path)])

    assert result == output_path.resolve()
    assert output_path.exists()
