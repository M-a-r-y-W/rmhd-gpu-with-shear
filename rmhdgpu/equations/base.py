"""Expected interface for equation modules.

Equation modules in this package are intentionally procedural. A typical module
should provide:

- `FIELD_NAMES`: ordered list of field names used by the system
- `rhs(state, ...)`: return the right-hand side as a `State`-like object
- `diagnostics(state, ...)`: return system-specific diagnostics
- `compute_conserved_quantity_budgets(state, ...)`: optionally return
  conserved-quantity values plus named non-conservative RHS terms
- `linear_matrix(grid, ...)`: return linear operator data for future use

This foundation does not impose an inheritance-heavy design.
"""
