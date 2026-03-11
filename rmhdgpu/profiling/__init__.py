"""Profiling and benchmarking helpers for rmhdgpu."""

from __future__ import annotations

from typing import Any


def run_backend_case(*args: Any, **kwargs: Any) -> Any:
    from rmhdgpu.profiling.benchmark_backends import run_backend_case as _run_backend_case

    return _run_backend_case(*args, **kwargs)


__all__ = ["run_backend_case"]
