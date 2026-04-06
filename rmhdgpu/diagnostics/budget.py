"""Helpers for conserved-quantity budget diagnostics.

Saved RHS terms use the sign convention

`d_t Q = sum(saved signed RHS terms)`

so dissipative contributions are negative when they remove the conserved
quantity and forcing contributions are positive when they inject it.
"""

from __future__ import annotations

from typing import Any


def flatten_conserved_quantity_budgets(
    budgets: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Flatten nested budget dictionaries to scalar CSV columns."""

    flat: dict[str, float] = {}
    for quantity_name, payload in budgets.items():
        flat[quantity_name] = float(payload.get("value", 0.0))
        rhs_terms = {name: float(value) for name, value in payload.get("rhs_terms", {}).items()}
        for term_name in sorted(rhs_terms):
            flat[f"{quantity_name}_rhs_{term_name}"] = rhs_terms[term_name]
        flat[f"{quantity_name}_rhs_total"] = float(sum(rhs_terms.values()))
    return flat
