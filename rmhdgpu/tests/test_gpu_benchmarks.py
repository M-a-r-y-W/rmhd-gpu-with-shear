from __future__ import annotations

import math

import pytest

from rmhdgpu.profiling.benchmark_backends import run_backend_case


cupy = pytest.importorskip("cupy")
try:
    cupy.zeros((1,), dtype=cupy.float64)
except Exception as exc:  # pragma: no cover - depends on runtime availability
    pytest.skip(f"CuPy is installed but not usable in this environment: {exc}", allow_module_level=True)


def test_cupy_short_run_executes() -> None:
    result = run_backend_case("cupy", 8, steps=2, dt=1.0e-3)

    assert result["steps"] == 2
    assert math.isfinite(float(result["elapsed_s"]))
    assert float(result["elapsed_s"]) > 0.0
    assert float(result["time_per_step_s"]) > 0.0


def test_cupy_backend_faster_than_numpy_smoke() -> None:
    numpy_result = run_backend_case("numpy", 8, steps=2, dt=1.0e-3)
    cupy_result = run_backend_case("cupy", 8, steps=2, dt=1.0e-3)

    for result in (numpy_result, cupy_result):
        assert math.isfinite(float(result["elapsed_s"]))
        assert float(result["elapsed_s"]) > 0.0
        assert math.isfinite(float(result["time_per_step_s"]))
        assert float(result["time_per_step_s"]) > 0.0
