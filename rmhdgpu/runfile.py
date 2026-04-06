"""Helpers for `.input` input files and resolved run settings."""

from __future__ import annotations

import argparse
import math
import tomllib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rmhdgpu.config import Config, default_config_dict


PRIMARY_INPUT_SUFFIX = ".input"
LEGACY_INPUT_SUFFIX = ".run"
ACCEPTED_INPUT_SUFFIXES = {PRIMARY_INPUT_SUFFIX, LEGACY_INPUT_SUFFIX}
DEFAULT_OUTPUT_DIR_NAME = "outputs"

_TOP_LEVEL_KEYS = {
    "title",
    "output_dir",
    "grid",
    "time",
    "output",
    "backend",
    "runtime",
    "physics",
    "forcing",
    "dissipation",
    "initial_condition",
}
_SECTION_KEYS = {
    "grid": {"Nx", "Ny", "Nz", "Lx", "Ly", "Lz"},
    "time": {
        "tmax",
        "dt_init",
        "dt_min",
        "dt_max",
        "cfl_number",
        "use_variable_dt",
        "t_out_scal",
        "t_out_spec",
        "t_out_full",
    },
    "output": {"t_out_scal", "t_out_spec", "t_out_full"},
    "backend": {"backend", "fft_workers", "real_dtype", "complex_dtype"},
    "runtime": {"runtime_check_every", "progress_output_every", "fail_on_nonfinite", "dealias", "dealias_mode"},
    "physics": {"vA", "cs2_over_vA2"},
    "forcing": {"use_forcing", "n_min_force", "n_max_force", "alpha_force", "forcing_seed", "force_amplitudes"},
    "initial_condition": {
        "type",
        "k_indices",
        "amplitude",
        "branch",
        "phi_seed",
        "phi_amplitude",
        "psi_seed",
        "psi_amplitude",
        "upar_seed",
        "upar_amplitude",
        "dbpar_seed",
        "dbpar_amplitude",
        "s_seed",
        "s_amplitude",
    },
}
_AUTO_DISSIPATION_KEYS = {
    "mode",
    "n_perp",
    "n_par",
    "nu_par",
    "kd_fraction",
    "shell_half_width",
    "update_every",
    "smooth_factor",
    "nu_min",
    "nu_max",
    "max_update_factor",
}
_SECTION_TO_CONFIG_KEYS = {
    "grid": {"Nx", "Ny", "Nz", "Lx", "Ly", "Lz"},
    "time": {"tmax", "dt_init", "dt_min", "dt_max", "cfl_number", "use_variable_dt", "t_out_scal", "t_out_spec", "t_out_full"},
    "output": {"t_out_scal", "t_out_spec", "t_out_full"},
    "backend": {"backend", "fft_workers", "real_dtype", "complex_dtype"},
    "runtime": {"runtime_check_every", "progress_output_every", "fail_on_nonfinite", "dealias", "dealias_mode"},
    "physics": {"vA", "cs2_over_vA2"},
    "forcing": {"use_forcing", "n_min_force", "n_max_force", "alpha_force", "forcing_seed"},
}


@dataclass(slots=True)
class InitialConditionSpec:
    """Driver-level initial-condition selection."""

    type: str = "alfven_mode"
    k_indices: tuple[int, int, int] = (1, 1, 1)
    amplitude: float = 1.0
    branch: str = "plus"
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, data: dict[str, Any] | None) -> "InitialConditionSpec":
        if data is None:
            return cls(parameters={})
        if not isinstance(data, dict):
            raise ValueError("initial_condition must be a TOML table.")

        unexpected = sorted(set(data) - _SECTION_KEYS["initial_condition"])
        if unexpected:
            raise ValueError(f"initial_condition has unexpected keys: {unexpected}.")

        kind = str(data.get("type", "alfven_mode"))
        raw_k = data.get("k_indices", [1, 1, 1])
        if not isinstance(raw_k, list) or len(raw_k) != 3:
            raise ValueError("initial_condition.k_indices must be an array of three integers.")
        k_indices = tuple(int(value) for value in raw_k)
        amplitude = float(data.get("amplitude", 1.0))
        branch = str(data.get("branch", "plus"))
        parameters = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "type",
                "k_indices",
                "amplitude",
                "branch",
            }
        }

        spec = cls(type=kind, k_indices=k_indices, amplitude=amplitude, branch=branch, parameters=parameters)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.parameters is None:
            self.parameters = {}
        if self.type not in {"alfven_mode", "zero", "aw_packet", "decaying_low_modes"}:
            raise ValueError(
                "initial_condition.type must be 'alfven_mode', 'zero', 'aw_packet', or "
                f"'decaying_low_modes'; got {self.type!r}."
            )
        if self.type == "alfven_mode":
            if len(self.k_indices) != 3:
                raise ValueError("initial_condition.k_indices must contain exactly three entries.")
            for index, value in enumerate(self.k_indices):
                if not isinstance(value, int):
                    raise ValueError(
                        f"initial_condition.k_indices[{index}] must be an integer; got {value!r}."
                    )
                if value < 0:
                    raise ValueError(
                        f"initial_condition.k_indices[{index}] must be nonnegative; got {value!r}."
                    )
            if self.amplitude <= 0.0:
                raise ValueError(f"initial_condition.amplitude must be positive; got {self.amplitude!r}.")
            if self.branch not in {"plus", "minus"}:
                raise ValueError(
                    f"initial_condition.branch must be 'plus' or 'minus'; got {self.branch!r}."
                )
        if self.type == "decaying_low_modes":
            for key, value in self.parameters.items():
                if key.endswith("_seed"):
                    int(value)
                if key.endswith("_amplitude") and float(value) <= 0.0:
                    raise ValueError(f"initial_condition.{key} must be positive; got {value!r}.")

    def to_document(self) -> dict[str, Any]:
        document = {
            "type": self.type,
        }
        if self.type == "alfven_mode":
            document["k_indices"] = list(self.k_indices)
            document["amplitude"] = self.amplitude
            document["branch"] = self.branch
        document.update(dict(self.parameters))
        return document


@dataclass(slots=True)
class RunSettings:
    """Resolved run settings after file loading and CLI override application."""

    config: Config
    output_dir: Path
    output_dir_setting: str
    title: str | None
    input_file: Path | None
    initial_condition: InitialConditionSpec
    resolved_document: dict[str, Any]


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _require_table(data: dict[str, Any], section: str) -> dict[str, Any]:
    value = data.get(section)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a TOML table.")
    return value


def load_run_file(path: str | Path) -> dict[str, Any]:
    """Load a `.input` file using TOML syntax."""

    runfile_path = Path(path).expanduser().resolve()
    if runfile_path.suffix not in ACCEPTED_INPUT_SUFFIXES:
        raise ValueError(
            "Run input files must end with "
            f"{PRIMARY_INPUT_SUFFIX!r} (legacy {LEGACY_INPUT_SUFFIX!r} is also accepted); "
            f"got {runfile_path.name!r}."
        )
    try:
        with runfile_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Could not parse TOML input file {runfile_path}: {exc}.") from exc
    except FileNotFoundError as exc:
        raise ValueError(f"Run input file not found: {runfile_path}.") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Run input file {runfile_path} did not parse to a TOML table.")

    unexpected_top_level = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unexpected_top_level:
        raise ValueError(f"Run input file {runfile_path} has unexpected top-level keys: {unexpected_top_level}.")
    if "title" in data and not isinstance(data["title"], str):
        raise ValueError("title must be a string when provided.")
    if "output_dir" in data and not isinstance(data["output_dir"], str):
        raise ValueError("output_dir must be a string when provided.")

    for section_name, allowed_keys in _SECTION_KEYS.items():
        section = _require_table(data, section_name)
        unexpected = sorted(set(section) - allowed_keys)
        if unexpected:
            raise ValueError(f"{section_name} has unexpected keys: {unexpected}.")

    forcing = _require_table(data, "forcing")
    force_amplitudes = forcing.get("force_amplitudes", {})
    if force_amplitudes is not None and not isinstance(force_amplitudes, dict):
        raise ValueError("forcing.force_amplitudes must be a TOML table.")

    dissipation = data.get("dissipation", {})
    if dissipation is not None and not isinstance(dissipation, dict):
        raise ValueError("dissipation must be a TOML table.")
    if isinstance(dissipation, dict):
        unexpected_scalar_keys = sorted(
            key for key, value in dissipation.items() if not isinstance(value, dict) and key not in _AUTO_DISSIPATION_KEYS
        )
        if unexpected_scalar_keys:
            raise ValueError(
                "dissipation has unexpected scalar keys: "
                f"{unexpected_scalar_keys}. Expected auto-mode keys are {sorted(_AUTO_DISSIPATION_KEYS)}."
            )
        for field_name, entry in dissipation.items():
            if isinstance(entry, dict):
                continue
            if field_name in _AUTO_DISSIPATION_KEYS:
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"dissipation.{field_name} must be a TOML table.")

    return data


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert explicitly provided CLI arguments into document-style overrides."""

    overrides: dict[str, Any] = {}
    values = vars(args)

    def _set_section(section: str, key: str, value: Any) -> None:
        overrides.setdefault(section, {})[key] = value

    if "title" in values:
        overrides["title"] = values["title"]
    if "output_dir" in values:
        overrides["output_dir"] = values["output_dir"]

    grid_map = {"nx": "Nx", "ny": "Ny", "nz": "Nz", "lx": "Lx", "ly": "Ly", "lz": "Lz"}
    for cli_key, config_key in grid_map.items():
        if cli_key in values:
            _set_section("grid", config_key, values[cli_key])

    time_map = {
        "tmax": "tmax",
        "dt_init": "dt_init",
        "dt_min": "dt_min",
        "dt_max": "dt_max",
        "cfl_number": "cfl_number",
        "use_variable_dt": "use_variable_dt",
    }
    for cli_key, config_key in time_map.items():
        if cli_key in values:
            _set_section("time", config_key, values[cli_key])

    output_map = {
        "t_out_scal": "t_out_scal",
        "t_out_spec": "t_out_spec",
        "t_out_full": "t_out_full",
    }
    for cli_key, config_key in output_map.items():
        if cli_key in values:
            _set_section("output", config_key, values[cli_key])

    backend_map = {"backend": "backend", "fft_workers": "fft_workers"}
    for cli_key, config_key in backend_map.items():
        if cli_key in values:
            _set_section("backend", config_key, values[cli_key])

    runtime_map = {
        "runtime_check_every": "runtime_check_every",
        "progress_output_every": "progress_output_every",
        "fail_on_nonfinite": "fail_on_nonfinite",
        "dealias": "dealias",
        "dealias_mode": "dealias_mode",
    }
    for cli_key, config_key in runtime_map.items():
        if cli_key in values:
            _set_section("runtime", config_key, values[cli_key])

    physics_map = {"vA": "vA", "cs2_over_vA2": "cs2_over_vA2"}
    for cli_key, config_key in physics_map.items():
        if cli_key in values:
            _set_section("physics", config_key, values[cli_key])

    forcing_map = {
        "use_forcing": "use_forcing",
        "forcing_seed": "forcing_seed",
        "n_min_force": "n_min_force",
        "n_max_force": "n_max_force",
        "alpha_force": "alpha_force",
    }
    for cli_key, config_key in forcing_map.items():
        if cli_key in values:
            _set_section("forcing", config_key, values[cli_key])
    if "force_sigma" in values:
        overrides.setdefault("forcing", {}).setdefault("force_amplitudes", {})
        overrides["forcing"]["force_amplitudes"]["psi"] = values["force_sigma"]
        overrides["forcing"]["force_amplitudes"]["omega"] = values["force_sigma"]

    initial_condition_map = {
        "initial_condition": "type",
        "mode_amplitude": "amplitude",
        "mode_branch": "branch",
    }
    for cli_key, config_key in initial_condition_map.items():
        if cli_key in values:
            _set_section("initial_condition", config_key, values[cli_key])
    mode_keys = ["mode_kx", "mode_ky", "mode_kz"]
    if any(key in values for key in mode_keys):
        current = overrides.setdefault("initial_condition", {})
        existing = list(current.get("k_indices", [1, 1, 1]))
        for index, key in enumerate(mode_keys):
            if key in values:
                existing[index] = values[key]
        current["k_indices"] = existing

    return overrides


def _apply_section_to_config_dict(config_values: dict[str, Any], section_name: str, section_data: dict[str, Any]) -> None:
    for key, value in section_data.items():
        if key in _SECTION_TO_CONFIG_KEYS.get(section_name, set()):
            config_values[key] = deepcopy(value)


def _document_to_config_values(document: dict[str, Any]) -> dict[str, Any]:
    config_values = default_config_dict()
    for section_name in ("grid", "time", "output", "backend", "runtime", "physics", "forcing"):
        section_data = _require_table(document, section_name)
        _apply_section_to_config_dict(config_values, section_name, section_data)

    forcing = _require_table(document, "forcing")
    force_amplitudes = forcing.get("force_amplitudes")
    if force_amplitudes is not None:
        config_values["force_amplitudes"].update(deepcopy(force_amplitudes))

    dissipation = document.get("dissipation")
    if dissipation is not None:
        for field_name, entry in dissipation.items():
            if isinstance(entry, dict):
                if field_name not in config_values["dissipation"]:
                    config_values["dissipation"][field_name] = {}
                config_values["dissipation"][field_name].update(deepcopy(entry))
            else:
                config_values["auto_dissipation"][field_name] = deepcopy(entry)

    return config_values


def _resolve_output_dir_setting(document: dict[str, Any]) -> str:
    output_dir = document.get("output_dir")
    if output_dir is None:
        return DEFAULT_OUTPUT_DIR_NAME
    return str(output_dir)


def _resolve_output_dir_path(output_dir_setting: str, *, input_file: Path | None, cwd: Path | None = None) -> Path:
    base_dir = input_file.parent if input_file is not None else Path.cwd() if cwd is None else Path(cwd)
    output_dir = Path(output_dir_setting).expanduser()
    if output_dir.is_absolute():
        return output_dir
    return (base_dir / output_dir).resolve()


def _resolved_document(
    *,
    config: Config,
    title: str | None,
    output_dir_setting: str,
    initial_condition: InitialConditionSpec,
) -> dict[str, Any]:
    config_values = deepcopy(default_config_dict())
    config_values.update(
        {
            "Nx": config.Nx,
            "Ny": config.Ny,
            "Nz": config.Nz,
            "Lx": config.Lx,
            "Ly": config.Ly,
            "Lz": config.Lz,
            "backend": config.backend,
            "fft_workers": config.fft_workers,
            "real_dtype": str(config.real_dtype),
            "complex_dtype": str(config.complex_dtype),
            "tmax": config.tmax,
            "dt_init": config.dt_init,
            "dt_min": config.dt_min,
            "dt_max": config.dt_max,
            "cfl_number": config.cfl_number,
            "use_variable_dt": config.use_variable_dt,
            "runtime_check_every": config.runtime_check_every,
            "progress_output_every": config.progress_output_every,
            "fail_on_nonfinite": config.fail_on_nonfinite,
            "t_out_scal": config.t_out_scal,
            "t_out_spec": config.t_out_spec,
            "t_out_full": config.t_out_full,
            "dealias": config.dealias,
            "dealias_mode": config.dealias_mode,
            "vA": config.vA,
            "cs2_over_vA2": config.cs2_over_vA2,
            "use_forcing": config.use_forcing,
            "n_min_force": config.n_min_force,
            "n_max_force": config.n_max_force,
            "alpha_force": config.alpha_force,
            "forcing_seed": config.forcing_seed,
            "force_amplitudes": deepcopy(config.force_amplitudes),
        "dissipation": deepcopy(config.dissipation),
        "auto_dissipation": asdict(config.auto_dissipation),
    }
    )
    return {
        "title": title,
        "output_dir": output_dir_setting,
        "grid": {
            "Nx": config_values["Nx"],
            "Ny": config_values["Ny"],
            "Nz": config_values["Nz"],
            "Lx": config_values["Lx"],
            "Ly": config_values["Ly"],
            "Lz": config_values["Lz"],
        },
        "time": {
            "tmax": config_values["tmax"],
            "dt_init": config_values["dt_init"],
            "dt_min": config_values["dt_min"],
            "dt_max": config_values["dt_max"],
            "cfl_number": config_values["cfl_number"],
            "use_variable_dt": config_values["use_variable_dt"],
        },
        "output": {
            "t_out_scal": config_values["t_out_scal"],
            "t_out_spec": config_values["t_out_spec"],
            "t_out_full": config_values["t_out_full"],
        },
        "backend": {
            "backend": config_values["backend"],
            "fft_workers": config_values["fft_workers"],
            "real_dtype": config_values["real_dtype"],
            "complex_dtype": config_values["complex_dtype"],
        },
        "runtime": {
            "runtime_check_every": config_values["runtime_check_every"],
            "progress_output_every": config_values["progress_output_every"],
            "fail_on_nonfinite": config_values["fail_on_nonfinite"],
            "dealias": config_values["dealias"],
            "dealias_mode": config_values["dealias_mode"],
        },
        "physics": {
            "vA": config_values["vA"],
            "cs2_over_vA2": config_values["cs2_over_vA2"],
        },
        "forcing": {
            "use_forcing": config_values["use_forcing"],
            "n_min_force": config_values["n_min_force"],
            "n_max_force": config_values["n_max_force"],
            "alpha_force": config_values["alpha_force"],
            "forcing_seed": config_values["forcing_seed"],
            "force_amplitudes": config_values["force_amplitudes"],
        },
        "dissipation": {
            **config_values["auto_dissipation"],
            **config_values["dissipation"],
        },
        "initial_condition": initial_condition.to_document(),
    }


def resolve_run_settings(
    *,
    runfile_path: str | Path | None,
    cli_overrides: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> RunSettings:
    """Resolve defaults, run-file values, and CLI overrides into final settings."""

    input_file = None if runfile_path is None else Path(runfile_path).expanduser().resolve()
    document = {} if input_file is None else load_run_file(input_file)
    if cli_overrides:
        document = _deep_merge(document, cli_overrides)

    config_values = _document_to_config_values(document)
    try:
        config = Config(**config_values)
    except ValueError as exc:
        source = f"run file {input_file}" if input_file is not None else "command line arguments"
        raise ValueError(f"Invalid configuration from {source}: {exc}.") from exc

    try:
        initial_condition = InitialConditionSpec.from_document(document.get("initial_condition"))
    except ValueError as exc:
        source = f"run file {input_file}" if input_file is not None else "command line arguments"
        raise ValueError(f"Invalid initial_condition from {source}: {exc}.") from exc
    title = None if document.get("title") is None else str(document["title"])
    output_dir_setting = _resolve_output_dir_setting(document)
    output_dir = _resolve_output_dir_path(output_dir_setting, input_file=input_file, cwd=cwd)

    return RunSettings(
        config=config,
        output_dir=output_dir,
        output_dir_setting=output_dir_setting,
        title=title,
        input_file=input_file,
        initial_condition=initial_condition,
        resolved_document=_resolved_document(
            config=config,
            title=title,
            output_dir_setting=output_dir_setting,
            initial_condition=initial_condition,
        ),
    )


def _toml_key_value(key: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"Cannot serialize null TOML value for key {key!r}.")
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    elif isinstance(value, float):
        if math.isfinite(value):
            rendered = repr(value)
        elif math.isnan(value):
            rendered = "nan"
        elif value > 0.0:
            rendered = "+inf"
        else:
            rendered = "-inf"
    elif isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        rendered = f'"{escaped}"'
    elif isinstance(value, list):
        rendered = "[" + ", ".join(_toml_value(item) for item in value) + "]"
    else:
        raise ValueError(f"Unsupported TOML value for key {key!r}: {value!r}.")
    return f"{key} = {rendered}"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0.0 else "-inf"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise ValueError(f"Unsupported TOML array value: {value!r}.")


def dump_toml(document: dict[str, Any]) -> str:
    """Serialize a nested mapping to a small TOML string."""

    lines: list[str] = []

    def _emit_table(table: dict[str, Any], prefix: list[str]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict) and value is not None]
        table_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]

        if prefix:
            lines.append("")
            lines.append(f"[{'.'.join(prefix)}]")

        for key, value in scalar_items:
            lines.append(_toml_key_value(key, value))

        for key, value in table_items:
            _emit_table(value, [*prefix, key])

    _emit_table(document, [])
    return "\n".join(lines).lstrip() + "\n"


def write_resolved_config(settings: RunSettings, path: str | Path) -> None:
    """Write the fully resolved run settings to TOML."""

    Path(path).write_text(dump_toml(settings.resolved_document), encoding="utf-8")
