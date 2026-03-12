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

## Running the code

The main script to run the code is `run.py`. This can be edited to change options or features (this will be updated to read a run file). 

From inside the main folder (where `Readme.md` lives):

```
python -m rmhdgpu.run --options
```

There are a few of example scripts, modelled on `run.py`, that can act as qualitative checks that everything is working (see `examples` folder). For example, a decaying turbulence test (from random, large-scale noise):

```
python -m rmhdgpu.examples.sanity_decay_spectra 
```
or on a GPU at 384^3, saving 20 frames to make a movie:
```
python -m rmhdgpu.examples.sanity_decay_spectra --t-final 8.0 --n 384 --backend cupy --save-frames --frame-count 20
```

You can use `-h` on tests and examples to return a list of the options available. 

## Profiling

The profiling utilities live under [`rmhdgpu/profiling`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling):

- [`benchmark_backends.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/benchmark_backends.py): compare `numpy`, `scipy_cpu`, and `cupy` on short representative runs
- [`profile_timestep.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/profile_timestep.py): coarse timing breakdown for FFTs, Poisson brackets, RHS work, forcing, diagnostics, and the whole step
- [`gpu_sanity.py`](/home/squjo23p/rmhd-gpu/rmhdgpu/profiling/gpu_sanity.py): verify that the CuPy backend is active, arrays stay on device, and short NumPy/CuPy runs agree within tolerance

Typical commands:

```bash
python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py rmhdgpu/tests/test_gpu_runtime_checks.py rmhdgpu/tests/test_gpu_benchmarks.py
python -m pytest
python -m rmhdgpu.profiling.benchmark_backends --backend numpy --backend scipy_cpu --backend cupy --nx 64 --nx 96 --steps 10
python -m rmhdgpu.profiling.profile_timestep --backend cupy --nx 64 --repeats 2
python -m rmhdgpu.profiling.gpu_sanity --nx 32 --steps 6
```

## Running on Aoraki GPUs

For this project, the most reliable setup on Aoraki is a user-local Conda environment. This avoids conflicts with the cluster’s system Python and makes the setup reproducible for students.

The main thing to remember is:

Do not use module load python for this project.

That module can override the Conda environment and leave you using /opt/spack/.../python even after conda activate, which usually shows up as missing imports such as numpy or cupy.

The recommended pattern is:

```bash
module purge
module load cuda
source ~/.bashrc
conda activate ~/conda-envs/curmpy
```

### Create the environment

Before doing this, you have to create a dedicated environment in your home directory:

```bash
mkdir -p ~/conda-envs
conda create -y -p ~/conda-envs/curmpy python=3.11
conda activate ~/conda-envs/curmpy
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pytest cupy
```

Then check that the environment is actually providing the Python interpreter:

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
python -c "import numpy, scipy, matplotlib, cupy; print('Environment OK')"
```

The which python and sys.executable outputs should point to something like

```bash
/home/<username>/conda-envs/curmpy/bin/python
```

not `/opt/spack/....`

### Optional activation helper

To avoid typing the full activation sequence every time, add a small helper script:

```
mkdir -p ~/bin
cat > ~/bin/activate-curmpy <<'EOF'
#!/usr/bin/env bash
source ~/.bashrc
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy
EOF
chmod +x ~/bin/activate-curmpy
```

You can also add an alias to `~/.bashrc`:

```
alias activate-curmpy="source ~/bin/activate-curmpy"
```

Then, in a new shell, you can just run:

```
activate-curmpy
```

### Interactive GPU workflow

A typical interactive workflow on an H100 node looks like this:

```bash
srun --partition=aoraki_gpu_H100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
module purge
module load cuda
source ~/.bashrc
conda activate ~/conda-envs/curmpy
cd ~/path/to/cuRMpy
```

If you use the helper script, the middle part becomes:

```
module purge
module load cuda
activate-curmpy
```

Once the environment is active, you can run tests or examples as usual. For example:

```
python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py
python -m rmhdgpu.examples.sanity_decay_spectra --gpu-256
python -m rmhdgpu.examples.sanity_forced_turbulence --gpu-256
```

If your account uses a different GPU partition, replace aoraki_gpu_H100 with the appropriate one, such as aoraki_gpu_A100_80GB.

### Batch / Slurm jobs

For batch jobs, activate the environment explicitly inside the job script. A minimal example is:

```
#!/bin/bash
#SBATCH --job-name=rmhdgpu
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load cuda
source ~/.bashrc
export PYTHONNOUSERSITE=1
conda activate ~/conda-envs/curmpy
cd ~/path/to/rmhd-gpu

python -m pytest rmhdgpu/tests/test_cupy_backend.py rmhdgpu/tests/test_gpu_consistency.py
python -m rmhdgpu.examples.sanity_decay_spectra --gpu
```

Submit it with: `sbatch run_rmhdgpu.slurm`

### Quick diagnostics

If something seems wrong, check which Python is actually active:

```
which python
python -V
python -c "import sys; print(sys.executable)"
python -c "import numpy, cupy; print(numpy.__version__, cupy.__version__)"
```

If which python points to /opt/spack/..., then the wrong Python is active and you are not actually using the Conda environment.

Common mistakes

The most common issues are:
- running module load python
- forgetting to activate the Conda environment in a fresh shell
- assuming the shell prompt alone proves the environment is correct
- using the system Python in a batch job instead of activating the environment explicitly

When in doubt, always check: `which python`

### Codex on Aoraki

Codex CLI can be run from an interactive Aoraki session in the same way as any other terminal tool. Run it only after the environment is activated, so imports, CuPy detection, and test behavior match the actual cluster run.
