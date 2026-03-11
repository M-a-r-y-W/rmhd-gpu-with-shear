"""Thin backend selection for NumPy, SciPy-FFT CPU, and CuPy."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Backend:
    """Small container that hides NumPy/CuPy selection details."""

    xp: Any
    backend_name: str
    is_gpu: bool
    fft_workers: int | None = None
    scipy_fft: Any | None = None

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def zeros_like(self, arr: Any) -> Any:
        return self.xp.zeros_like(arr)

    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.xp.ones(shape, dtype=dtype)

    def asarray(self, arr: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(arr, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Return a NumPy ndarray regardless of the active backend."""

        if self.is_gpu:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)

    def scalar_to_float(self, value: Any) -> float:
        """Convert a scalar reduction result into a Python float."""

        if self.is_gpu and hasattr(value, "item"):
            return float(value.item())
        return float(np.asarray(value))

    def scalar_to_int(self, value: Any) -> int:
        """Convert a scalar reduction result into a Python int."""

        if self.is_gpu and hasattr(value, "item"):
            return int(value.item())
        return int(np.asarray(value))

    def random_generator(self, seed: int | None = None) -> Any:
        """Return a backend-native random generator when available."""

        default_rng = getattr(self.xp.random, "default_rng", None)
        if default_rng is not None:
            return default_rng(seed)
        if self.is_gpu:
            return self.xp.random.RandomState(seed)
        return np.random.default_rng(seed)

    def synchronize(self) -> None:
        """Synchronize outstanding device work for timing-sensitive paths."""

        if not self.is_gpu:
            return

        runtime = getattr(getattr(self.xp, "cuda", None), "runtime", None)
        if runtime is not None and hasattr(runtime, "deviceSynchronize"):
            runtime.deviceSynchronize()
            return

        stream_cls = getattr(getattr(self.xp, "cuda", None), "Stream", None)
        if stream_cls is not None and hasattr(stream_cls, "null"):
            stream_cls.null.synchronize()


def build_backend(config: Any) -> Backend:
    """Build the requested array backend from a configuration object."""

    if config.backend == "numpy":
        return Backend(
            xp=np,
            backend_name="numpy",
            is_gpu=False,
            fft_workers=config.fft_workers,
            scipy_fft=None,
        )

    if config.backend == "scipy_cpu":
        try:
            scipy_fft = importlib.import_module("scipy.fft")
        except ImportError as exc:
            raise ImportError(
                "Config requested backend='scipy_cpu', but SciPy FFT support could not be imported."
            ) from exc
        return Backend(
            xp=np,
            backend_name="scipy_cpu",
            is_gpu=False,
            fft_workers=config.fft_workers,
            scipy_fft=scipy_fft,
        )

    if config.backend == "cupy":
        try:
            cp = importlib.import_module("cupy")
        except ImportError as exc:
            raise ImportError(
                "Config requested backend='cupy', but CuPy could not be imported."
            ) from exc
        return Backend(
            xp=cp,
            backend_name="cupy",
            is_gpu=True,
            fft_workers=None,
            scipy_fft=None,
        )

    raise ValueError(f"Unsupported backend {config.backend!r}.")
