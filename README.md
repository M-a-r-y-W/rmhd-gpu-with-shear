# rmhdgpu

`rmhdgpu` is a single-node Fourier pseudo-spectral RMHD-style solver with three backend modes:

- `numpy`: baseline CPU path using `numpy.fft`
- `scipy_cpu`: CPU path using `scipy.fft`
- `cupy`: single-GPU path using `cupy.fft`

Current solver scope includes:

- ideal homogeneous S09 equations
- anisotropic dissipation with integrating-factor time stepping
- variable timestep support
- stochastic forcing
- scalar and spectral diagnostics
- NumPy, SciPy CPU, and CuPy backends
- formal tests plus lightweight profiling and sanity scripts

## Backend Notes

- The CuPy backend keeps Fourier and workspace arrays on device during evolution.
- Scalar diagnostics and runtime finite checks only transfer scalar reductions back to the host.
- Spectra and full-field outputs intentionally return NumPy arrays because they are usually consumed by plotting or I/O code.
- Forcing uses backend-native random generation when available. A fixed seed is reproducible within a backend, but NumPy and CuPy runs are not expected to be bitwise identical.

## Tests

CPU-focused tests live under [`rmhdgpu/tests`](/home/squjo23p/rmhd-gpu/rmhdgpu/tests).

GPU-focused tests skip cleanly when CuPy is unavailable or when a usable CUDA device is not present:

- [`test_cupy_backend.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/tests/test_cupy_backend.py)
- [`test_gpu_consistency.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/tests/test_gpu_consistency.py)
- [`test_gpu_runtime_checks.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/tests/test_gpu_runtime_checks.py)
- [`test_gpu_benchmarks.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/tests/test_gpu_benchmarks.py)

## Profiling

The profiling utilities live under [`rmhdgpu/profiling`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling):

- [`benchmark_backends.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/benchmark_backends.py): compare `numpy`, `scipy_cpu`, and `cupy` on short representative runs
- [`profile_timestep.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/profile_timestep.py): coarse timing breakdown for FFTs, Poisson brackets, RHS work, forcing, diagnostics, and the whole step
- [`gpu_sanity.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/gpu_sanity.py): verify that the CuPy backend is active, arrays stay on device, and short NumPy/CuPy runs agree within tolerance

Typical commands:

```bash
python3 -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py rmhdgpu/tests/test_gpu_runtime_checks.py rmhdgpu/tests/test_gpu_benchmarks.py
python3 -m pytest
python3 -m rmhdgpu.profiling.benchmark_backends --backend numpy --backend scipy_cpu --backend cupy --nx 64 --nx 96 --steps 10
python3 -m rmhdgpu.profiling.profile_timestep --backend cupy --nx 64 --repeats 2
python3 -m rmhdgpu.profiling.gpu_sanity --nx 32 --steps 6
```
