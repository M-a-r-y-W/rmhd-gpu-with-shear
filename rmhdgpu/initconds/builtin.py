"""Built-in initial conditions and the lightweight initcond registry.

This module is the single source of truth for input-file selectable initial
conditions. Add a new built-in initial condition by:

1. writing a builder with signature
   `builder(*, parameters, grid, backend, fft, dealias_mask, field_names, params) -> State`
2. writing a small parameter normalizer for that builder
3. registering the builder with `@register_initial_condition("name", ...)`

`rmhdgpu.run` and the example scripts both dispatch through
`build_initial_state(...)` so every registered name follows the same path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rmhdgpu.initconds.eigenmodes_s09 import alfven_mode_state
from rmhdgpu.masks import apply_mask
from rmhdgpu.operators import dx, dy, lap_perp
from rmhdgpu.state import State
from rmhdgpu.equations import get_equation_module


InitialConditionBuilder = Callable[..., State]
ParameterNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


DECAY_LOW_MODE_DEFAULTS = {
    "phi_seed": 1,
    "phi_amplitude": 0.4,
    "psi_seed": 2,
    "psi_amplitude": 0.3,
    "upar_seed": 3,
    "upar_amplitude": 0.08,
    "dbpar_seed": 4,
    "dbpar_amplitude": 0.06,
    "s_seed": 5,
    "s_amplitude": 0.05,
    "a_seed": 6,
    "a_amplitude": 0.05,
}


@dataclass(frozen=True, slots=True)
class InitialConditionDefinition:
    """Registered initial-condition metadata."""

    name: str
    builder: InitialConditionBuilder
    normalize_parameters: ParameterNormalizer
    description: str


_REGISTRY: dict[str, InitialConditionDefinition] = {}


def register_initial_condition(
    name: str,
    *,
    normalize_parameters: ParameterNormalizer | None = None,
    description: str = "",
) -> Callable[[InitialConditionBuilder], InitialConditionBuilder]:
    """Register a named initial-condition builder."""

    registry_name = str(name)

    def decorator(builder: InitialConditionBuilder) -> InitialConditionBuilder:
        if registry_name in _REGISTRY:
            raise ValueError(f"Initial condition {registry_name!r} is already registered.")
        _REGISTRY[registry_name] = InitialConditionDefinition(
            name=registry_name,
            builder=builder,
            normalize_parameters=_identity_parameters if normalize_parameters is None else normalize_parameters,
            description=description,
        )
        return builder

    return decorator


def list_initial_condition_types() -> list[str]:
    """Return the registered input-file initial-condition names."""

    return sorted(_REGISTRY)


def get_initial_condition_builder(name: str) -> InitialConditionBuilder:
    """Return the registered builder for `name`."""

    return _get_initial_condition_definition(name).builder


def normalize_initial_condition_parameters(name: str, parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize parameters for a registered initial condition."""

    definition = _get_initial_condition_definition(name)
    raw_parameters = _as_parameter_dict(parameters)
    try:
        return definition.normalize_parameters(raw_parameters)
    except ValueError as exc:
        raise ValueError(f"Invalid parameters for initial condition {name!r}: {exc}") from exc


def build_initial_state(
    initial_condition: Any,
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an initial state from a registered name or spec-like object."""

    kind, raw_parameters = _coerce_initial_condition_request(initial_condition, parameters=parameters)
    normalized_parameters = normalize_initial_condition_parameters(kind, raw_parameters)
    builder = get_initial_condition_builder(kind)
    return builder(
        parameters=normalized_parameters,
        grid=grid,
        backend=backend,
        fft=fft,
        dealias_mask=dealias_mask,
        field_names=field_names,
        params=params,
    )


def low_mode_real_field(
    grid: Any,
    backend: Any,
    *,
    seed: int,
    amplitude: float,
) -> Any:
    """Return a smooth low-mode random real field."""

    rng = np.random.default_rng(seed)
    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = backend.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                a_cos = rng.normal(scale=amplitude / 6.0)
                a_sin = rng.normal(scale=amplitude / 6.0)
                phase = nx * x + ny * y + nz * z
                field += a_cos * xp.cos(phase) + a_sin * xp.sin(phase)

    return field.astype(grid.real_dtype, copy=False)


def aw_packet_real_field(grid: Any, backend: Any) -> Any:
    """Return the large-amplitude Alfvénic packet used by the examples."""

    xp = backend.xp
    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    field = xp.zeros(grid.real_shape, dtype=grid.real_dtype)

    for nx in range(1, 4):
        for ny in range(1, 4):
            for nz in range(1, 4):
                coefficient = 0.15 / (nx + ny + nz - 1.0)
                phase = 0.3 * (nx - ny + nz)
                field += coefficient * xp.cos(nx * x + ny * y + nz * z + phase)

    rms = backend.scalar_to_float(xp.sqrt(xp.mean(field**2)))
    field *= 1.0 / rms
    return field


def initial_u_rms(phi_hat: Any, grid: Any, fft: Any, backend: Any) -> float:
    """Return the perpendicular RMS velocity associated with `phi_hat`."""

    ux = -fft.c2r(dy(phi_hat, grid))
    uy = fft.c2r(dx(phi_hat, grid))
    xp = backend.xp
    return backend.scalar_to_float(xp.sqrt(xp.mean(ux**2 + uy**2)))


def _identity_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return dict(parameters)


def _get_initial_condition_definition(name: str) -> InitialConditionDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_initial_condition_types())
        raise ValueError(
            f"Unknown initial condition type {name!r}. Available initial conditions: {available}."
        ) from exc


def _as_parameter_dict(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ValueError("initial-condition parameters must be a mapping.")
    return {str(key): value for key, value in parameters.items()}


def _coerce_initial_condition_request(
    initial_condition: Any,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if isinstance(initial_condition, str):
        return initial_condition, _as_parameter_dict(parameters)

    if isinstance(initial_condition, Mapping):
        if "type" not in initial_condition:
            raise ValueError("initial_condition mappings must contain a 'type' entry.")
        merged = {
            key: value
            for key, value in initial_condition.items()
            if key not in {"type", "parameters"}
        }
        merged.update(_as_parameter_dict(initial_condition.get("parameters")))
        merged.update(_as_parameter_dict(parameters))
        return str(initial_condition["type"]), merged

    if hasattr(initial_condition, "type"):
        merged = _as_parameter_dict(getattr(initial_condition, "parameters", None))
        merged.update(_as_parameter_dict(parameters))
        return str(initial_condition.type), merged

    raise TypeError(
        "initial_condition must be a registered name, a mapping with 'type', "
        "or an object with 'type' and 'parameters' attributes."
    )


def _reject_unknown_parameters(kind: str, parameters: dict[str, Any], allowed: set[str]) -> None:
    unexpected = sorted(set(parameters) - allowed)
    if unexpected:
        raise ValueError(f"unexpected parameters {unexpected}; allowed parameters are {sorted(allowed)}.")


def _require_fields(kind: str, field_names: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in field_names]
    if missing:
        raise ValueError(
            f"Initial condition {kind!r} requires fields {list(required)!r}; missing {missing!r}."
        )


def _masked_r2c(real_field: Any, *, fft: Any, dealias_mask: Any | None) -> Any:
    field_hat = fft.r2c(real_field)
    if dealias_mask is not None:
        apply_mask(field_hat, dealias_mask)
    return field_hat


def _normalize_zero_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("zero", parameters, set())
    return {}


def _normalize_aw_packet_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("aw_packet", parameters, set())
    return {}


def _normalize_alfven_mode_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"k_indices", "amplitude", "branch"}
    _reject_unknown_parameters("alfven_mode", parameters, allowed)

    raw_k = parameters.get("k_indices", [1, 1, 1])
    if not isinstance(raw_k, (list, tuple)) or len(raw_k) != 3:
        raise ValueError("k_indices must be an array of three integers.")

    k_indices: list[int] = []
    for index, value in enumerate(raw_k):
        if not isinstance(value, int):
            raise ValueError(f"k_indices[{index}] must be an integer; got {value!r}.")
        if value < 0:
            raise ValueError(f"k_indices[{index}] must be nonnegative; got {value!r}.")
        k_indices.append(int(value))

    amplitude = float(parameters.get("amplitude", 1.0))
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive; got {amplitude!r}.")

    branch = str(parameters.get("branch", "plus"))
    if branch not in {"plus", "minus"}:
        raise ValueError(f"branch must be 'plus' or 'minus'; got {branch!r}.")

    return {
        "k_indices": k_indices,
        "amplitude": amplitude,
        "branch": branch,
    }


def _normalize_low_beta_mode_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"k_indices", "amplitude", "mode"}
    _reject_unknown_parameters("low_beta_stratified_mode", parameters, allowed)

    raw_k = parameters.get("k_indices", [0, 1, 0])
    if not isinstance(raw_k, (list, tuple)) or len(raw_k) != 3:
        raise ValueError("k_indices must be an array of three integers.")
    k_indices: list[int] = []
    for index, value in enumerate(raw_k):
        if not isinstance(value, int):
            raise ValueError(f"k_indices[{index}] must be an integer; got {value!r}.")
        if value < 0:
            raise ValueError(f"k_indices[{index}] must be nonnegative; got {value!r}.")
        k_indices.append(int(value))

    amplitude = float(parameters.get("amplitude", 1.0))
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive; got {amplitude!r}.")

    mode = str(parameters.get("mode", "unstable_growing"))
    allowed_modes = {"unstable_growing", "unstable_decaying", "stable_plus", "stable_minus"}
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}; got {mode!r}.")

    return {"k_indices": k_indices, "amplitude": amplitude, "mode": mode}


def _normalize_decaying_low_modes_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_parameters("decaying_low_modes", parameters, set(DECAY_LOW_MODE_DEFAULTS))

    normalized = dict(DECAY_LOW_MODE_DEFAULTS)
    normalized.update(parameters)

    for key in list(normalized):
        if key.endswith("_seed"):
            normalized[key] = int(normalized[key])
        if key.endswith("_amplitude"):
            normalized[key] = float(normalized[key])
            if normalized[key] <= 0.0:
                raise ValueError(f"{key} must be positive; got {normalized[key]!r}.")

    return normalized


@register_initial_condition(
    "zero",
    normalize_parameters=_normalize_zero_parameters,
    description="All-zero Fourier state.",
)
def zero(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an all-zero initial state."""

    _normalize_zero_parameters(_as_parameter_dict(parameters))
    return State(grid, backend, field_names=list(field_names))


@register_initial_condition(
    "alfven_mode",
    normalize_parameters=_normalize_alfven_mode_parameters,
    description="Exact linear Alfvén eigenmode in Fourier space.",
)
def alfven_mode(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build an exact linear Alfvén eigenmode."""

    normalized = _normalize_alfven_mode_parameters(_as_parameter_dict(parameters))
    _require_fields("alfven_mode", field_names, ("psi", "omega"))
    return alfven_mode_state(
        grid=grid,
        backend=backend,
        field_names=list(field_names),
        k_indices=normalized["k_indices"],
        amplitude=normalized["amplitude"],
        branch=normalized["branch"],
        params=params,
    )


@register_initial_condition(
    "low_beta_stratified_mode",
    normalize_parameters=_normalize_low_beta_mode_parameters,
    description="Single linear eigenmode of the low_beta_stratified equation set.",
)
def low_beta_stratified_mode(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build a single linear eigenmode using the equation module eigensystem."""

    normalized = _normalize_low_beta_mode_parameters(_as_parameter_dict(parameters))
    _require_fields("low_beta_stratified_mode", field_names, ("psi", "omega", "a"))
    equation_module = get_equation_module("low_beta_stratified")

    ix_raw, iy_raw, iz = normalized["k_indices"]
    if iz < 0 or iz > grid.Nz // 2:
        raise ValueError(f"k_indices[2] must satisfy 0 <= kz <= Nz//2; got {iz}.")
    ix = ix_raw % grid.Nx
    iy = iy_raw % grid.Ny
    kx = backend.scalar_to_float(grid.kx[ix, 0, 0])
    ky = backend.scalar_to_float(grid.ky[0, iy, 0])
    kz = backend.scalar_to_float(grid.kz[0, 0, iz])

    matrix = equation_module.linear_matrix(kx=kx, ky=ky, kz=kz, params=params)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    mode = normalized["mode"]
    if mode == "unstable_growing":
        selected = int(np.argmax(eigenvalues.real))
    elif mode == "unstable_decaying":
        selected = int(np.argmin(eigenvalues.real))
    elif mode == "stable_plus":
        selected = int(np.argmax(eigenvalues.imag))
    else:
        selected = int(np.argmin(eigenvalues.imag))

    vector = eigenvectors[:, selected]
    scale = np.max(np.abs(vector))
    if scale <= 0.0:
        raise ValueError("Selected low-beta eigenvector has zero amplitude.")
    vector = normalized["amplitude"] * vector / scale

    state = State(grid, backend, field_names=list(field_names))
    for component, field_name in enumerate(equation_module.FIELD_NAMES):
        state[field_name][ix, iy, iz] = vector[component]
    if dealias_mask is not None:
        state.apply_mask(dealias_mask)
    return state


@register_initial_condition(
    "aw_packet",
    normalize_parameters=_normalize_aw_packet_parameters,
    description="Large-amplitude nonlinear Alfvénic packet used by the examples.",
)
def aw_packet(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build the large-amplitude Alfvénic packet example state."""

    _normalize_aw_packet_parameters(_as_parameter_dict(parameters))
    _require_fields("aw_packet", field_names, ("psi", "omega"))

    phi_hat = _masked_r2c(aw_packet_real_field(grid, backend), fft=fft, dealias_mask=dealias_mask)
    state = State(grid, backend, field_names=list(field_names))
    state["psi"][...] = phi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)
    return state


@register_initial_condition(
    "decaying_low_modes",
    normalize_parameters=_normalize_decaying_low_modes_parameters,
    description="Low-mode random multifield initial condition used by the decaying examples.",
)
def decaying_low_modes(
    *,
    parameters: Mapping[str, Any] | None = None,
    grid: Any,
    backend: Any,
    fft: Any,
    dealias_mask: Any | None,
    field_names: Sequence[str],
    params: Any,
) -> State:
    """Build the multifield low-mode random initial condition."""

    normalized = _normalize_decaying_low_modes_parameters(_as_parameter_dict(parameters))
    _require_fields("decaying_low_modes", field_names, ("psi", "omega"))

    state = State(grid, backend, field_names=list(field_names))
    phi_hat = _masked_r2c(
        low_mode_real_field(
            grid,
            backend,
            seed=normalized["phi_seed"],
            amplitude=normalized["phi_amplitude"],
        ),
        fft=fft,
        dealias_mask=dealias_mask,
    )
    psi_hat = _masked_r2c(
        low_mode_real_field(
            grid,
            backend,
            seed=normalized["psi_seed"],
            amplitude=normalized["psi_amplitude"],
        ),
        fft=fft,
        dealias_mask=dealias_mask,
    )

    state["psi"][...] = psi_hat
    state["omega"][...] = lap_perp(phi_hat, grid)

    for field_name, seed_key, amplitude_key in (
        ("upar", "upar_seed", "upar_amplitude"),
        ("dbpar", "dbpar_seed", "dbpar_amplitude"),
        ("s", "s_seed", "s_amplitude"),
        ("a", "a_seed", "a_amplitude"),
    ):
        if field_name not in state.field_names:
            continue
        state[field_name][...] = _masked_r2c(
            low_mode_real_field(
                grid,
                backend,
                seed=normalized[seed_key],
                amplitude=normalized[amplitude_key],
            ),
            fft=fft,
            dealias_mask=dealias_mask,
        )

    return state
