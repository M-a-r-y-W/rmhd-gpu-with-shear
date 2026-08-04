"""Minimal helpers for extracting full real-space fields."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def extract_full_fields(
    state: Any,
    fft: Any,
    backend: Any,
    field_names: Iterable[str] | None = None,
    extra_fields_hat: Mapping[str,Any] | None = None,
) -> dict[str, Any]:
    """Inverse transform requested fields and return NumPy arrays."""

    names = list(field_names) if field_names is not None else state.field_names
    fields = {name: backend.to_numpy(fft.c2r(state[name])) for name in names}

    if extra_fields_hat:
        for name, field_hat in extra_fields_hat.items():
            fields[name] = backend.to_numpy(fft.c2r(field_hat))

    return fields

