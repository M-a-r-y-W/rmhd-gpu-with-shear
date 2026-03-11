"""Small utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rmhdgpu.errors import NonFiniteStateError


def max_abs_finite(arr: Any, backend: Any) -> float:
    """Return the maximum absolute value over the finite entries of `arr`."""

    finite_mask = backend.xp.isfinite(arr)
    finite_count = backend.scalar_to_int(backend.xp.count_nonzero(finite_mask))
    if finite_count == 0:
        return float("nan")
    finite_abs = backend.xp.where(finite_mask, backend.xp.abs(arr), 0.0)
    return backend.scalar_to_float(backend.xp.max(finite_abs))


def check_state_finite(
    state: Any,
    backend: Any | None = None,
    *,
    time: float | None = None,
    step: int | None = None,
    context: str = "",
) -> None:
    """Raise `NonFiniteStateError` if any field contains NaN or infinite values."""

    backend_obj = state.backend if backend is None else backend
    failures: list[tuple[str, int, float]] = []

    for name in state.field_names:
        arr = state[name]
        nonfinite_mask = ~backend_obj.xp.isfinite(arr)
        nonfinite_count = backend_obj.scalar_to_int(backend_obj.xp.count_nonzero(nonfinite_mask))
        if nonfinite_count > 0:
            failures.append((name, nonfinite_count, max_abs_finite(arr, backend_obj)))

    if not failures:
        return

    location_parts: list[str] = []
    if step is not None:
        location_parts.append(f"step {step}")
    if time is not None:
        location_parts.append(f"t={time:.6g}")
    location = ", ".join(location_parts)
    if location:
        location = f" at {location}"

    context_suffix = f" during {context}" if context else ""
    field_names = ", ".join(name for name, _, _ in failures)
    first_name, first_count, first_max_abs = failures[0]
    raise NonFiniteStateError(
        "Non-finite values detected in field(s) "
        f"{field_names}{location}{context_suffix}. "
        f"First failing field '{first_name}' has {first_count} non-finite entries; "
        f"max |finite| before failure was {first_max_abs:.6g}. "
        "This usually indicates numerical instability, for example a CFL number "
        "that is too large or dissipation settings that are unsuitable."
    )
