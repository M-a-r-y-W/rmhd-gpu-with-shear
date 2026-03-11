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

## Running on Aoraki GPUs

Use a user-local Conda or Miniforge environment rather than the system Python or site modules. That keeps the CuPy stack isolated, makes student setups reproducible, and avoids accidental mixing with `~/.local` packages.

Create an environment in a fixed user path:

```bash
mkdir -p ~/conda-envs
conda create -y -p ~/conda-envs/curmpy python=3.11
conda activate ~/conda-envs/curmpy
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pytest cupy
```

Add a small activation helper so you do not need to remember the full Conda path each time:

```bash
mkdir -p ~/bin
cat > ~/bin/activate-curmpy <<'EOF'
#!/usr/bin/env bash
source ~/miniforge3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy
EOF
chmod +x ~/bin/activate-curmpy
```

Typical interactive GPU workflow:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
source ~/bin/activate-curmpy
cd ~/path/to/rmhd-gpu
python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py
python -m rmhdgpu.examples.sanity_decay_spectra --gpu-256
python -m rmhdgpu.examples.sanity_forced_turbulence --gpu-256
```

If your Aoraki account uses a different GPU partition name, replace `gpu` in the `srun` command.

Minimal batch-job skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=rmhdgpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/miniforge3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy
cd ~/path/to/rmhd-gpu

python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py
python -m rmhdgpu.examples.sanity_decay_spectra --gpu-256
```

Submit that with:

```bash
sbatch run_rmhdgpu.slurm
```

Codex CLI can be run from an interactive Aoraki session in the same way as any other terminal tool. Run it only after the environment is activated, and test the project inside that activated environment so imports and CuPy detection match the actual cluster run.

## Example Movie Frames

The three main sanity scripts can optionally write x-y midplane PNG sequences of vorticity and current for later movie assembly:

```bash
python -m rmhdgpu.examples.sanity_aw_packet --save-frames --frame-count 12
python -m rmhdgpu.examples.sanity_decay_spectra --save-frames --frame-count 12
python -m rmhdgpu.examples.sanity_forced_turbulence --save-frames --frame-count 12
```

Use `--snapshot-z-index` to choose a different plane. Frames are written into `frames_vorticity/` and `frames_current/` inside the chosen output directory, with fixed symmetric color limits per run so the resulting movie does not rescale from frame to frame.
