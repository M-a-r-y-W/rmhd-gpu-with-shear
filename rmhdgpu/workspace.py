"""Reusable scratch buffers for hot operator routines."""

from __future__ import annotations

from typing import Any


class Workspace:
    """Preallocate named real and complex scratch buffers."""

    def __init__(
        self,
        grid: Any,
        backend: Any,
        n_real_buffers: int = 6,
        n_complex_buffers: int = 4,
    ) -> None:
        self.grid = grid
        self.backend = backend
        self.real = {
            f"r{i}": backend.zeros(grid.real_shape, dtype=grid.real_dtype)
            for i in range(n_real_buffers)
        }
        self.complex = {
            f"c{i}": backend.zeros(grid.fourier_shape, dtype=grid.complex_dtype)
            for i in range(n_complex_buffers)
        }
        self.states: dict[str, Any] = {}
        self.cache: dict[Any, Any] = {}

    def get_state_buffer(self, key: str, field_names: Any) -> Any:
        """Return a reusable scratch state with the requested field layout."""

        from rmhdgpu.state import State

        names = list(field_names)
        state = self.states.get(key)
        if state is None:
            state = State(self.grid, self.backend, field_names=names)
            self.states[key] = state
            return state

        if state.field_names != names:
            raise ValueError(
                f"Workspace state buffer {key!r} was initialized for fields "
                f"{state.field_names!r}, not {names!r}."
            )
        return state
