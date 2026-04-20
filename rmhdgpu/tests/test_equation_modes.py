from __future__ import annotations

import numpy as np

from rmhdgpu.backend import build_backend
from rmhdgpu.config import Config
from rmhdgpu.equations import s09
from rmhdgpu.fft import FFTManager
from rmhdgpu.grid import build_grid
from rmhdgpu.operators import lap_perp
from rmhdgpu.run import _zero_poisson_bracket, main
from rmhdgpu.state import State
from rmhdgpu.workspace import Workspace


def test_zero_poisson_bracket_makes_ideal_rhs_linear(monkeypatch) -> None:
    config = Config(Nx=8, Ny=8, Nz=8, backend="numpy", dealias=False)
    backend = build_backend(config)
    grid = build_grid(config, backend)
    fft = FFTManager(grid, backend)
    workspace = Workspace(grid, backend)

    x = grid.x.reshape(grid.Nx, 1, 1)
    y = grid.y.reshape(1, grid.Ny, 1)
    z = grid.z.reshape(1, 1, grid.Nz)
    phi_real = np.cos(x) + 0.4 * np.cos(2.0 * y) + 0.2 * np.cos(x + y) + 0.0 * z
    phi_hat = fft.r2c(phi_real.astype(grid.real_dtype, copy=False))

    state = State(grid, backend, field_names=config.field_names)
    state["omega"][...] = lap_perp(phi_hat, grid)

    nonlinear = s09.ideal_rhs(state, grid, fft, workspace, config)
    monkeypatch.setattr(s09, "poisson_bracket", _zero_poisson_bracket)
    linearized = s09.ideal_rhs(state, grid, fft, workspace, config)

    for name in state.field_names:
        np.testing.assert_allclose(backend.to_numpy(linearized[name]), 0.0, atol=1.0e-13)
    assert np.max(np.abs(backend.to_numpy(nonlinear["omega"]))) > 1.0e-8


def test_run_linear_mode_calls_ideal_rhs_with_zero_poisson_bracket(tmp_path, monkeypatch) -> None:
    input_file = tmp_path / "linear.input"
    input_file.write_text(
        """
output_dir = "outputs"

[equations]
type = "s09"
mode = "linear"

[grid]
Nx = 8
Ny = 8
Nz = 8

[time]
tmax = 0.001
dt_init = 0.001
use_variable_dt = false

[output]
t_out_scal = 0.001
t_out_spec = 0.0
t_out_full = 0.0

[runtime]
progress_output_every = 100
""".strip()
        + "\n",
        encoding="utf-8",
    )

    original_poisson_bracket = s09.poisson_bracket
    original_ideal_rhs = s09.ideal_rhs
    call_count = 0

    def tracking_ideal_rhs(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        assert s09.poisson_bracket is _zero_poisson_bracket
        return original_ideal_rhs(*args, **kwargs)

    monkeypatch.setattr(s09, "ideal_rhs", tracking_ideal_rhs)

    main([str(input_file)])

    assert call_count > 0
    assert s09.poisson_bracket is original_poisson_bracket
    assert (tmp_path / "outputs" / "scalar_diagnostics.csv").exists()
