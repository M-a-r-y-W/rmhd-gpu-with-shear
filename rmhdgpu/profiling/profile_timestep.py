"""Coarse timing breakdown for one timestep or a short run."""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from typing import Any, Iterator

from rmhdgpu.diagnostics.scalar import compute_scalar_diagnostics
from rmhdgpu.equations import s09
from rmhdgpu.forcing import apply_forcing_kick, generate_forcing_kick
from rmhdgpu.profiling.benchmark_backends import build_benchmark_case
from rmhdgpu.steppers import if_ssprk3_step


class TimingAccumulator:
    """Collect coarse wall-clock timings for named regions."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self.backend.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.backend.synchronize()
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1


class TimedFFTManager:
    """Wrap an FFT manager and attribute time to forward/inverse transforms."""

    def __init__(self, fft: Any, timings: TimingAccumulator) -> None:
        self._fft = fft
        self._timings = timings
        self.grid = fft.grid
        self.backend = fft.backend
        self.xp = fft.xp

    def r2c(self, f_real: Any, out: Any | None = None) -> Any:
        with self._timings.measure("fft_r2c"):
            return self._fft.r2c(f_real, out=out)

    def c2r(self, f_hat: Any, out: Any | None = None) -> Any:
        with self._timings.measure("fft_c2r"):
            return self._fft.c2r(f_hat, out=out)


def profile_timestep(
    backend_name: str,
    nx: int,
    *,
    dt: float,
    use_forcing: bool,
    include_diagnostics: bool,
    repeats: int,
) -> dict[str, float | int | str]:
    config, backend, grid, fft_base, workspace, mask, state = build_benchmark_case(
        backend_name,
        nx,
        dt=dt,
        use_forcing=use_forcing,
    )
    timings = TimingAccumulator(backend)
    fft = TimedFFTManager(fft_base, timings)
    linear_ops = s09.build_dissipation_operators(grid, config)
    rhs_kwargs = {
        "grid": grid,
        "fft": fft,
        "workspace": workspace,
        "params": config,
        "dealias_mask": mask,
    }
    forcing_rng = backend.random_generator(config.forcing_seed) if use_forcing else None

    original_poisson_bracket = s09.poisson_bracket
    original_ideal_rhs = s09.ideal_rhs

    def timed_poisson_bracket(*args: Any, **kwargs: Any) -> Any:
        with timings.measure("poisson_bracket"):
            return original_poisson_bracket(*args, **kwargs)

    def timed_ideal_rhs(*args: Any, **kwargs: Any) -> Any:
        with timings.measure("ideal_rhs"):
            return original_ideal_rhs(*args, **kwargs)

    s09.poisson_bracket = timed_poisson_bracket
    current = state
    try:
        for _ in range(repeats):
            with timings.measure("whole_step"):
                current = if_ssprk3_step(
                    current,
                    dt,
                    timed_ideal_rhs,
                    linear_ops,
                    rhs_kwargs=rhs_kwargs,
                )
                if use_forcing:
                    with timings.measure("forcing"):
                        forcing_kick = generate_forcing_kick(
                            current,
                            grid,
                            fft,
                            backend,
                            config,
                            forcing_rng,
                            dt,
                            workspace=workspace,
                            out=workspace.get_state_buffer("forcing_kick", current.field_names),
                        )
                        current = apply_forcing_kick(current, forcing_kick, inplace=True)
                if include_diagnostics:
                    with timings.measure("diagnostics"):
                        compute_scalar_diagnostics(current, grid, fft, backend, workspace=workspace)
    finally:
        s09.poisson_bracket = original_poisson_bracket
        s09.ideal_rhs = original_ideal_rhs

    totals = dict(timings.totals)
    totals.update({f"{name}_count": count for name, count in timings.counts.items()})
    totals["backend"] = backend_name
    totals["nx"] = nx
    totals["repeats"] = repeats
    return totals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy", choices=["numpy", "scipy_cpu", "cupy"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--use-forcing", action="store_true")
    parser.add_argument("--include-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = profile_timestep(
        args.backend,
        args.nx,
        dt=args.dt,
        use_forcing=args.use_forcing,
        include_diagnostics=args.include_diagnostics,
        repeats=args.repeats,
    )
    for key in sorted(result):
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
