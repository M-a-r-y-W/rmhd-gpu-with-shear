"""Thin wrappers around `rfftn` / `irfftn`."""

from __future__ import annotations

from functools import partial
from typing import Any


class FFTManager:
    """Wrap backend FFT calls with one consistent convention.

    Backend behavior:

    - `numpy`: NumPy arrays with `numpy.fft`
    - `scipy_cpu`: NumPy arrays with `scipy.fft`, optionally using
      `backend.fft_workers`
    - `cupy`: CuPy arrays with `cupy.fft`

    All backends use the same default FFT normalization:

    - forward transform: unnormalized
    - inverse transform: scaled by `1 / (Nx * Ny * Nz)`

    With that convention, `c2r(r2c(f))` recovers the original real-space field
    to numerical precision when the Fourier array is compatible with `rfftn`
    storage.
    """

    def __init__(self, grid: Any, backend: Any) -> None:
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.axes = (0, 1, 2)
        self.real_shape = grid.real_shape

        if self.backend.backend_name == "scipy_cpu":
            self._rfftn = partial(
                self.backend.scipy_fft.rfftn,
                s=self.real_shape,
                axes=self.axes,
                workers=self.backend.fft_workers,
            )
            self._irfftn = partial(
                self.backend.scipy_fft.irfftn,
                s=self.real_shape,
                axes=self.axes,
                workers=self.backend.fft_workers,
            )
        else:
            self._rfftn = partial(self.xp.fft.rfftn, s=self.real_shape, axes=self.axes)
            self._irfftn = partial(self.xp.fft.irfftn, s=self.real_shape, axes=self.axes)

    def r2c(self, f_real: Any, out: Any | None = None) -> Any:
        """Transform a real field to `rfftn` Fourier storage."""

        transformed = self._rfftn(f_real)
        if out is not None:
            out[...] = transformed
            return out
        return transformed

    def c2r(self, f_hat: Any, out: Any | None = None) -> Any:
        """Transform an `rfftn` Fourier field back to real space."""

        transformed = self._irfftn(f_hat)
        if out is not None:
            out[...] = transformed
            return out
        return transformed
