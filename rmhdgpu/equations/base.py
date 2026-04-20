"""Expected interface for equation modules.

Equation modules in this package are intentionally procedural. A typical module
should provide:

- `FIELD_NAMES`: ordered list of field names used by the system
- `derived_parameters(params)`: return the compact scalar parameter block
- `ideal_rhs(state, ...)`: return the nondissipative right-hand side
- `build_dissipation_operators(grid, params, ...)`: return diagonal damping data
- `total_energy(state, ...)`: return the equation-set energy diagnostic
- `compute_conserved_quantity_budgets(state, ...)`: optionally return
  conserved-quantity values plus named non-conservative RHS terms
- `linear_matrix(kx, ky, kz, params)`: return linear operator data for one mode

This foundation does not impose an inheritance-heavy design.
"""
