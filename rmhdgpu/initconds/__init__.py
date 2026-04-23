"""Unified initial-condition registry and helper utilities.

`initial_condition.type` values from `.input` files map to builders registered
here. To add a new built-in initial condition, register a builder with
`register_initial_condition(...)` and keep its parameter normalization close to
the implementation so the supported inputs remain obvious to inspect.
"""

from rmhdgpu.initconds.builtin import (
    DECAY_LOW_MODE_DEFAULTS,
    alfven_mode,
    aw_packet,
    aw_packet_real_field,
    build_initial_state,
    decaying_low_modes,
    get_initial_condition_builder,
    initial_u_rms,
    list_initial_condition_types,
    low_mode_real_field,
    low_beta_stratified_mode,
    normalize_initial_condition_parameters,
    register_initial_condition,
    zero,
)
from rmhdgpu.initconds.eigenmodes_s09 import (
    alfven_mode_state,
    entropy_mode_state,
    slow_mode_state,
)
from rmhdgpu.initconds.eigenmodes_low_beta_stratified import low_beta_stratified_mode_state
from rmhdgpu.initconds.testing import single_mode_field

__all__ = [
    "DECAY_LOW_MODE_DEFAULTS",
    "alfven_mode",
    "alfven_mode_state",
    "aw_packet",
    "aw_packet_real_field",
    "build_initial_state",
    "decaying_low_modes",
    "entropy_mode_state",
    "get_initial_condition_builder",
    "initial_u_rms",
    "list_initial_condition_types",
    "low_mode_real_field",
    "low_beta_stratified_mode",
    "low_beta_stratified_mode_state",
    "normalize_initial_condition_parameters",
    "register_initial_condition",
    "single_mode_field",
    "slow_mode_state",
    "zero",
]
