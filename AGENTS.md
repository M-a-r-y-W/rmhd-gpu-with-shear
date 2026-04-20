# AGENTS.md

## Project overview

This repository contains a Fourier pseudo-spectral RMHD-style solver for homogeneous and related reduced systems, with support for:

- `numpy` backend
- `scipy_cpu` backend for local multicore FFTs
- `cupy` backend for single-GPU runs on NVIDIA hardware

The code is intended to be:

- scientifically correct
- robust on CPU and GPU
- readable and easy for students to modify
- practical for case-based simulation workflows on local machines and remote clusters

Prefer small, explicit changes over clever abstractions.

---

## Main priorities

When modifying this repository, prioritize the following, in order:

1. **Correctness**
2. **Robustness**
3. **Student usability / readability**
4. **Performance**
5. **Convenience features**

Do not trade away correctness or clarity for a small performance gain unless explicitly asked.

---

## Critical scientific expectations

The following are especially important and must not be broken casually:

- **Correct linear behavior** for single-mode tests and eigenmode-based tests
- **Correct ideal conservation laws**, especially energy conservation
- **Consistent energy-budget diagnostics**
- **Correct dissipation and forcing bookkeeping**
- **Correct Fourier normalization and operator conventions**
- **Correct handling of the physical amplitudes** (`u_perp ~ grad_perp phi`, `b_perp ~ grad_perp psi`) rather than raw potentials where relevant

Any change touching:
- equations
- dissipation
- forcing
- energy definitions
- budget diagnostics
- timestepper logic
- backend-specific execution

should come with targeted tests.

---

## Solver architecture

Keep the code structure broadly as follows:

- `rmhdgpu/` contains the solver core
- `rmhdgpu/equations/` contains equation-set-specific physics
- `rmhdgpu/initconds/` is the single source of truth for initial conditions
- `rmhdgpu/examples/` contains runnable example/sanity scripts
- `vis/` contains general post-processing scripts that read saved outputs
- `.input` files are the normal user-facing run configuration mechanism
- `rmhdgpu.run` is the main run driver

Do not create duplicate mechanisms when one clear mechanism already exists.

Examples:
- initial conditions should live under `rmhdgpu.initconds`
- visualization should work off saved outputs, not hidden solver internals
- new run controls should go through the `.input` / Config path unless there is a strong reason otherwise

---

## Code style and design preferences

### General style

- Prefer **procedural, explicit code**
- Keep functions readable and physics-facing where possible
- Avoid heavy object-oriented frameworks
- Avoid unnecessary abstraction
- Avoid introducing new dependencies unless genuinely useful

### Comments and documentation

- README should stay **practical**
- Detailed implementation reasoning should go in:
  - docstrings
  - comments near the relevant functions
  - tests, when appropriate

When changing physics-facing code, comments should explain:
- what is being computed
- which normalization is being used
- any important sign conventions
- why the implementation is structured that way

### Adding features

When adding a feature:
- make it work in the normal `.input`-driven workflow
- make it save useful outputs if relevant
- make it testable
- keep it understandable for a good student reading the code for the first time

---

## Backends

The code supports:
- `numpy`
- `scipy_cpu`
- `cupy`

Backend compatibility matters.

### Expectations

- Keep large arrays on device when using `cupy`
- Avoid unnecessary host/device transfers
- Avoid recomputing expensive quantities unless needed
- Avoid introducing CPU-only logic in hot GPU paths unless explicitly justified
- Only transfer arrays to CPU at output / plotting / file-writing boundaries, or when scalar reductions are needed

### Do not break

- `numpy` correctness
- `scipy_cpu` local usability
- `cupy` single-GPU usability

For GPU-related changes:
- add skip-cleanly tests if CuPy is unavailable
- synchronize correctly when benchmarking or timing GPU work

---

## Running the code

The normal run workflow is:

python -m rmhdgpu.run case.input

or with overrides:

python -m rmhdgpu.run case.input –tmax 10.0 –backend cupy

CLI-only mode should also continue to work:

python -m rmhdgpu.run –backend numpy –nx 64 –tmax 0.1

For real runs, prefer case directories with:

- an .input file
- an outputs/ directory written by the code

When modifying the run workflow:

- keep file-based runs as the main user-facing path
- keep CLI overrides working
- save a resolved configuration for reproducibility

## Outputs and diagnostics

The code saves diagnostics in a reusable form. This behavior is important.

Current expectations:

- scalar diagnostics: CSV
- spectra: CSV
- full fields: HDF5
- visualization scripts should work off these saved outputs

When adding new diagnostics:

- save them in a format usable by vis/
- keep naming stable and grep-friendly
- include enough metadata to interpret them later
- avoid computing expensive diagnostics every step unless truly needed

If a quantity matters scientifically, it should usually be:

- saved
- documented
- testable

## Visualization scripts

The vis/ folder contains general post-processing scripts.

These scripts should:

- work on saved outputs from any simulation
- not depend on a specific example script
- be usable from both command line and Spyder
- support --show and sensible save behavior where practical

Do not hard-code assumptions that only work for one example unless the script is explicitly example-specific.

## Initial conditions

rmhdgpu.initconds is the single source of truth for initial-condition builders.

Rules:

- initial conditions selected by .input files should resolve through rmhdgpu.initconds
- adding a new initial condition should mean adding/registering it there
- do not duplicate initial-condition systems in example helper files

If an initial condition is used in examples, it should still be a proper registered initcond.

## Testing expectations

### General rule

Any change that affects the solver numerically should have tests.

Prefer:

- small
- deterministic
- targeted
- fast

tests.

Keep separate

- formal pytest tests
- larger sanity-check example scripts
- profiling scripts

Do not turn large visual sanity checks into heavyweight formal tests.

### Especially important tests

Changes to any of these areas should trigger targeted tests:

- equations and RHS
- timestepper
- linear modes
- ideal conservation
- dissipation
- forcing
- energy/budget diagnostics
- backend consistency
- run-file / input parsing

If in doubt, add a test.

## Performance guidance

Performance matters, but must not come at the expense of clarity unless explicitly requested.

### Good optimizations

Good optimizations include:

- reuse preallocated workspace
- avoid repeated allocations in hot loops
- avoid unnecessary FFTs
- avoid unnecessary CPU/GPU transfers
- avoid recomputing spectra or diagnostics unless due
- use Fourier-space quantities directly when possible

### Avoid by default

Avoid by default:

- major refactors just for speed
- custom kernels unless truly necessary
- opaque optimization tricks that make the code harder to maintain

### When optimizing:

1. verify correctness first
2. profile
3. optimize the actual bottleneck
4. keep the result readable

## Aoraki / cluster usage

This code is intended to run on Aoraki GPUs.

### Important practical rule:

- Do not use module load python for this project when running through the project Conda environment.

### Preferred Aoraki workflow:

- load cuda
- activate the project Conda environment
- run from there

Cluster-specific documentation lives in the README. Keep it practical and correct.

Do not introduce assumptions that break:

- interactive GPU runs
- batch/Slurm runs
- user-local Conda environments

## README expectations

The README should begin with the normal user workflow, not with developer-only testing details.

The README should clearly cover:

- what the code is
- how to run a normal case
- .input file workflow
- outputs
- basic visualization
- cluster setup (especially Aoraki) in practical terms

Deeper implementation details belong in code comments or tests, not at the top of the README.

### When making substantial changes

For substantial changes, a good workflow is:

1. inspect the current implementation
2. explain the planned change briefly
3. implement the change
4. run the most relevant targeted tests
5. summarize:
    - what changed
    - what was tested
    - any caveats or uncertainties

Be honest about uncertainty. Do not claim correctness without testing where testing is feasible.

### Practical “done” criteria

A change is usually not finished until:

- the code path is clear
- the relevant tests pass
- outputs are still sensible
- README or help text is updated if user-facing behavior changed
- no redundant/obsolete paths are left behind

If a refactor removes a mechanism, delete or simplify the old one rather than leaving confusing dead code behind.

## Repository-specific reminders

- Easy for students to use is a core design goal.
- Energy conservation and linear behavior are crucial.
- Budget diagnostics must remain trustworthy.
- Fourier-space conventions should not be changed casually.
- Keep example scripts useful as sanity checks, but do not let them become the solver architecture.
- Prefer one clean mechanism over two overlapping mechanisms.

