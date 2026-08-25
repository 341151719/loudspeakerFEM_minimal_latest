"""Pure-standard-library validation for phase-1 enclosure demonstrators.

This module deliberately validates declarations only.  It does not import
NumPy, Gmsh, a FEM solver, or any thermoviscous implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


class EnclosureSchemaError(ValueError):
    """Raised when an enclosure configuration cannot be used safely."""


# Short alias for callers that use the conventional schema terminology.
SchemaError = EnclosureSchemaError


ALLOWED_CASES = frozenset(
    {
        "base_axisym",
        "open_back",
        "sealed_lossless",
        "sealed_thermoviscous",
        "vented_rear_coaxial",
        "passive_radiator_rear_coaxial",
    }
)

_UNIT_SUFFIXES = (
    "_N_s_m3",
    "_N_s_m2",
    "_N_s_m",
    "_Pa_s_m3",
    "_Pa_s_m2",
    "_Pa_s_m",
    "_J_kgK",
    "_W_mK",
    "_kg_mol",
    "_kg_m3",
    "_m_s",
    "_m2",
    "_m3",
    "_kg",
    "_Hz",
    "_Pa",
    "_K",
    "_m",
    "_N",
    "_W",
    "_V",
    "_A",
    "_s",
)

# Numeric values in this set are dimensionless bookkeeping or applicability
# limits, not hidden SI quantities.
_DIMENSIONLESS_NUMERIC_KEYS = frozenset(
    {
        "adaptive_peak_points",
        "boundary_layer_to_gap_max",
        "boundary_layer_to_radius_max",
        "end_correction_factor",
        "element_order",
        "frequency_count",
        "gamma",
        "ka_max",
        "max_mach",
        "pml_element_count",
        "prandtl",
        "reference_fraction",
        "relative_tolerance",
    }
)


@dataclass(frozen=True)
class VolumeContract:
    gross_internal_volume_m3: float
    driver_displacement_m3: float
    port_displacement_m3: float
    passive_radiator_back_displacement_m3: float
    reserved_rear_feature_m3: float
    computed_net_volume_m3: float


@dataclass(frozen=True)
class EnclosureConfig:
    """Validated, immutable summary plus a copied raw declaration."""

    raw: dict[str, Any]
    case: str
    demonstrator: bool
    net_volume_target_m3: float
    computed_net_volume_m3: float
    volume_contract: VolumeContract


ValidatedEnclosureConfig = EnclosureConfig


def _path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _fail(path: str, message: str) -> None:
    raise EnclosureSchemaError(f"{path}: {message}")


def _walk_json(value: Any, path: str = "") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            _fail(path or "config", "numeric values must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path or "config", "keys must be strings")
            _walk_json(item, _path(path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_json(item, f"{path}[{index}]")
        return
    _fail(path or "config", f"unsupported value type {type(value).__name__}")


def _check_unit_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = _path(path, key)
            if (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and key not in _DIMENSIONLESS_NUMERIC_KEYS
                and not any(key.endswith(suffix) for suffix in _UNIT_SUFFIXES)
            ):
                _fail(item_path, "numeric SI parameters require a unit suffix")
            _check_unit_keys(item, item_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_unit_keys(item, f"{path}[{index}]")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        _fail(_path(path, key), "is required")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        _fail(_path(path, key), "must be a non-empty string")
    return value


def _boolean(mapping: Mapping[str, Any], key: str, path: str) -> bool:
    value = _required(mapping, key, path)
    if not isinstance(value, bool):
        _fail(_path(path, key), "must be boolean")
    return value


def _number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(_path(path, key), "must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(_path(path, key), "must be finite")
    return result


def _positive(mapping: Mapping[str, Any], key: str, path: str) -> float:
    result = _number(mapping, key, path)
    if result <= 0.0:
        _fail(_path(path, key), "must be > 0")
    return result


def _nonnegative(mapping: Mapping[str, Any], key: str, path: str) -> float:
    result = _number(mapping, key, path)
    if result < 0.0:
        _fail(_path(path, key), "must be >= 0")
    return result


def _choice(mapping: Mapping[str, Any], key: str, choices: set[str], path: str) -> str:
    value = _string(mapping, key, path)
    if value not in choices:
        _fail(_path(path, key), f"unsupported value {value!r}")
    return value


def _validate_provenance(value: Any) -> None:
    data = _mapping(value, "provenance")
    _string(data, "upstream_commit", "provenance")
    _string(data, "config_version", "provenance")
    _string(data, "generated_by", "provenance")
    _string(data, "parameter_source", "provenance")
    if _boolean(data, "measured", "provenance"):
        _fail("provenance.measured", "must be false for a demonstrator")
    assumptions = _required(data, "assumptions", "provenance")
    if not isinstance(assumptions, list) or not assumptions or not all(
        isinstance(item, str) and item.strip() for item in assumptions
    ):
        _fail("provenance.assumptions", "must be a non-empty list of text assumptions")
    section_sources = _mapping(
        _required(data, "section_sources", "provenance"), "provenance.section_sources"
    )
    for section in ("air", "geometry", "volume", "port", "passive_radiator", "thermoviscous"):
        _string(section_sources, section, "provenance.section_sources")


def _validate_physics(value: Any) -> None:
    data = _mapping(value, "physics")
    if _string(data, "harmonic_convention", "physics") != "exp(+i*omega*t)":
        _fail("physics.harmonic_convention", "must be exp(+i*omega*t)")
    if _string(data, "phasor_amplitude", "physics") != "peak":
        _fail("physics.phasor_amplitude", "must be peak")
    if _string(data, "normal_convention", "physics") != "acoustic_domain_to_structure":
        _fail("physics.normal_convention", "must be acoustic_domain_to_structure")
    if _string(data, "pressure_volume_velocity", "physics") != "p_Pa_over_U_m3_s":
        _fail("physics.pressure_volume_velocity", "must define p/U")
    if _string(data, "mechanical_acoustic_conversion", "physics") != "Zm_N_s_m_times_Sd_m2_squared":
        _fail("physics.mechanical_acoustic_conversion", "must define Zm = Za Sd^2")


def _validate_air(value: Any) -> None:
    data = _mapping(value, "air")
    for key in (
        "rho0_kg_m3",
        "c0_m_s",
        "p_ref_Pa",
        "temperature_K",
        "pressure_Pa",
        "dynamic_viscosity_Pa_s",
        "heat_capacity_cp_J_kgK",
        "thermal_conductivity_W_mK",
    ):
        _positive(data, key, "air")
    for key in ("gamma", "prandtl"):
        _positive(data, key, "air")


def _validate_geometry(value: Any) -> dict[str, float | bool | None]:
    data = _mapping(value, "geometry")
    if not _boolean(data, "axisymmetric", "geometry"):
        _fail("geometry.axisymmetric", "phase-1 enclosure cases must be axisymmetric")
    if _string(data, "shape", "geometry") != "cylindrical_demonstrator":
        _fail("geometry.shape", "must identify the demonstrator geometry")
    inner_radius = _positive(data, "inner_radius_m", "geometry")
    inner_depth = _positive(data, "inner_depth_m", "geometry")
    wall = _positive(data, "wall_thickness_m", "geometry")
    outer_radius = _positive(data, "outer_radius_m", "geometry")
    outer_depth = _positive(data, "outer_depth_m", "geometry")
    if outer_radius <= inner_radius or outer_depth <= inner_depth:
        _fail("geometry", "outer dimensions must exceed inner dimensions")
    if wall >= min(inner_radius, inner_depth) / 2.0:
        _fail("geometry.wall_thickness_m", "is too large for the cavity")
    driver_radius = _positive(data, "driver_radius_m", "geometry")
    if driver_radius >= inner_radius:
        _fail("geometry.driver_radius_m", "driver geometry does not fit the cavity")
    driver_projection = _positive(data, "driver_projection_length_m", "geometry")
    port_penetration = _nonnegative(data, "port_penetration_into_box_m", "geometry")
    rear_mount_radius = _positive(data, "rear_mount_radius_m", "geometry")
    if rear_mount_radius > outer_radius:
        _fail("geometry.rear_mount_radius_m", "exceeds the rear wall")
    rear_opening_enabled = _boolean(data, "rear_opening_enabled", "geometry")
    opening = data.get("rear_opening_radius_m")
    if rear_opening_enabled:
        if opening is None or isinstance(opening, bool) or not isinstance(opening, (int, float)):
            _fail("geometry.rear_opening_radius_m", "is required for an open rear feature")
        opening = float(opening)
        if not math.isfinite(opening) or opening <= 0.0 or opening > rear_mount_radius:
            _fail("geometry.rear_opening_radius_m", "is outside the rear wall")
    elif opening is not None:
        _fail("geometry.rear_opening_radius_m", "must be null when no rear opening is enabled")
    if not _boolean(data, "rear_feature_coaxial", "geometry"):
        _fail("geometry.rear_feature_coaxial", "must be true for the axisymmetric reference")
    pml_inner = _positive(data, "pml_inner_radius_m", "geometry")
    pml_thickness = _positive(data, "pml_thickness_m", "geometry")
    return {
        "inner_radius_m": inner_radius,
        "inner_depth_m": inner_depth,
        "wall_thickness_m": wall,
        "outer_radius_m": outer_radius,
        "outer_depth_m": outer_depth,
        "driver_radius_m": driver_radius,
        "driver_projection_length_m": driver_projection,
        "port_penetration_into_box_m": port_penetration,
        "rear_mount_radius_m": rear_mount_radius,
        "rear_opening_enabled": rear_opening_enabled,
        "rear_opening_radius_m": opening,
        "pml_inner_radius_m": pml_inner,
        "pml_thickness_m": pml_thickness,
    }


def _validate_volume(value: Any, target: float, geometry: Mapping[str, Any]) -> VolumeContract:
    data = _mapping(value, "volume_contract")
    gross = _positive(data, "gross_internal_volume_m3", "volume_contract")
    driver = _nonnegative(data, "driver_displacement_m3", "volume_contract")
    port = _nonnegative(data, "port_displacement_m3", "volume_contract")
    pr = _nonnegative(
        data, "passive_radiator_back_displacement_m3", "volume_contract"
    )
    reserved = _nonnegative(data, "reserved_rear_feature_m3", "volume_contract")
    computed = gross - driver - port - pr - reserved
    if computed <= 0.0:
        _fail("volume_contract", "computed net volume must be > 0")
    if not math.isclose(target, computed, rel_tol=1.0e-8, abs_tol=1.0e-12):
        _fail(
            "net_volume_target_m3",
            f"does not equal occupancy contract result {computed:.12g} m^3",
        )
    geometric = math.pi * float(geometry["inner_radius_m"]) ** 2 * float(
        geometry["inner_depth_m"]
    )
    if not math.isclose(gross, geometric, rel_tol=1.0e-6, abs_tol=1.0e-12):
        _fail("volume_contract.gross_internal_volume_m3", "does not match axisymmetric cavity volume")
    return VolumeContract(gross, driver, port, pr, reserved, computed)


def _validate_port(value: Any) -> tuple[Mapping[str, Any], bool]:
    data = _mapping(value, "port")
    enabled = _boolean(data, "enabled", "port")
    model = _string(data, "model", "port")
    if enabled and model != "lumped_helmholtz_reference":
        _fail("port.model", "enabled port must declare the lumped Helmholtz reference")
    if not enabled and model != "none":
        _fail("port.model", "disabled port must be none")
    cross_section = _choice(data, "cross_section", {"circular", "rectangular", "slit"}, "port")
    _positive(data, "radius_m", "port")
    _positive(data, "length_m", "port")
    _nonnegative(data, "resistance_Pa_s_m3", "port")
    _nonnegative(data, "surface_roughness_m", "port")
    _choice(
        data,
        "terminal_condition",
        {"open-open", "closed-open", "open-closed", "closed-closed"},
        "port",
    )
    loss_model = _choice(
        data,
        "loss_model",
        {
            "none",
            "BLI_boundary_operator_declared",
            "circular_pipe_lrf_declared",
            "narrow_region_lrf_declared",
        },
        "port",
    )
    if loss_model == "circular_pipe_lrf_declared" and cross_section != "circular":
        _fail("port.loss_model", "circular pipe loss requires a circular cross-section")
    if loss_model == "narrow_region_lrf_declared" and cross_section != "slit":
        _fail("port.loss_model", "narrow-region loss requires a slit cross-section")
    lumped = _mapping(_required(data, "lumped_reference", "port"), "port.lumped_reference")
    _nonnegative(lumped, "end_correction_factor", "port.lumped_reference")
    if _string(lumped, "applies_to", "port.lumped_reference") != "lumped_reference_only":
        _fail("port.lumped_reference.applies_to", "must not apply to explicit FEM")
    explicit = _mapping(_required(data, "explicit_fem", "port"), "port.explicit_fem")
    explicit_enabled = _boolean(explicit, "enabled", "port.explicit_fem")
    if _boolean(explicit, "implemented_in_phase1", "port.explicit_fem"):
        _fail("port.explicit_fem.implemented_in_phase1", "phase 1 cannot implement FEM")
    _string(explicit, "radiation_model", "port.explicit_fem")
    reuse_end = _boolean(explicit, "reuse_end_correction", "port.explicit_fem")
    reuse_mass = _boolean(explicit, "reuse_radiation_mass", "port.explicit_fem")
    reuse_resistance = _boolean(explicit, "reuse_radiation_resistance", "port.explicit_fem")
    if explicit_enabled and reuse_end:
        _fail("port.explicit_fem.reuse_end_correction", "would double-count end correction")
    if explicit_enabled and reuse_mass:
        _fail("port.explicit_fem.reuse_radiation_mass", "would double-count radiation mass")
    if explicit_enabled and reuse_resistance:
        _fail("port.explicit_fem.reuse_radiation_resistance", "would double-count radiation resistance")
    applicability = _mapping(_required(data, "applicability", "port"), "port.applicability")
    f_min = _positive(applicability, "frequency_min_Hz", "port.applicability")
    f_max = _positive(applicability, "frequency_max_Hz", "port.applicability")
    if f_max <= f_min:
        _fail("port.applicability", "frequency range is empty")
    _positive(applicability, "first_longitudinal_mode_Hz", "port.applicability")
    _positive(applicability, "minimum_radius_for_reference_m", "port.applicability")
    _positive(applicability, "boundary_layer_to_radius_max", "port.applicability")
    _positive(applicability, "ka_max", "port.applicability")
    return data, enabled


def _validate_pr(value: Any) -> tuple[Mapping[str, Any], bool]:
    data = _mapping(value, "passive_radiator")
    enabled = _boolean(data, "enabled", "passive_radiator")
    model = _string(data, "model", "passive_radiator")
    if enabled and model != "rigid_piston_sdof":
        _fail("passive_radiator.model", "enabled PR must be rigid_piston_sdof")
    if not enabled and model != "none":
        _fail("passive_radiator.model", "disabled PR must be none")
    _positive(data, "Sd_m2", "passive_radiator")
    _positive(data, "Mms_kg", "passive_radiator")
    _positive(data, "Cms_m_N", "passive_radiator")
    resistance = _positive(data, "Rms_N_s_m", "passive_radiator")
    if resistance <= 0.0:  # Keeps the error contract explicit if the rule changes.
        _fail("passive_radiator.Rms_N_s_m", "damping must be > 0 in a demonstrator")
    _positive(data, "rear_clearance_m", "passive_radiator")
    radiation = _choice(
        data,
        "radiation_model",
        {"none", "explicit_fem_outer_air_domain"},
        "passive_radiator",
    )
    includes_mass = _boolean(data, "Mms_includes_radiation", "passive_radiator")
    includes_resistance = _boolean(data, "Rms_includes_radiation", "passive_radiator")
    if radiation == "explicit_fem_outer_air_domain" and (includes_mass or includes_resistance):
        _fail("passive_radiator", "Mms/Rms must exclude FEM radiation loading")
    applicability = _mapping(
        _required(data, "applicability", "passive_radiator"),
        "passive_radiator.applicability",
    )
    f_min = _positive(applicability, "frequency_min_Hz", "passive_radiator.applicability")
    f_max = _positive(applicability, "frequency_max_Hz", "passive_radiator.applicability")
    if f_max <= f_min:
        _fail("passive_radiator.applicability", "frequency range is empty")
    return data, enabled


def _validate_thermoviscous(value: Any) -> None:
    data = _mapping(value, "thermoviscous")
    enabled = _boolean(data, "enabled", "thermoviscous")
    model = _choice(
        data,
        "model",
        {"none", "BLI_boundary_operator_declared", "narrow_region_lrf_declared"},
        "thermoviscous",
    )
    if enabled != (model != "none"):
        _fail("thermoviscous", "enabled flag and model choice conflict")
    if _boolean(data, "implemented_in_phase1", "thermoviscous"):
        _fail("thermoviscous.implemented_in_phase1", "phase 1 only declares applicability")
    _string(data, "cross_section_scope", "thermoviscous")
    applicability = _mapping(_required(data, "applicability", "thermoviscous"), "thermoviscous.applicability")
    f_min = _positive(applicability, "frequency_min_Hz", "thermoviscous.applicability")
    f_max = _positive(applicability, "frequency_max_Hz", "thermoviscous.applicability")
    if f_max <= f_min:
        _fail("thermoviscous.applicability", "frequency range is empty")
    _positive(applicability, "minimum_gap_m", "thermoviscous.applicability")
    _positive(applicability, "boundary_layer_to_gap_max", "thermoviscous.applicability")
    _positive(applicability, "boundary_layer_to_radius_max", "thermoviscous.applicability")
    if not _boolean(applicability, "boundary_layers_must_not_overlap", "thermoviscous.applicability"):
        _fail("thermoviscous.applicability.boundary_layers_must_not_overlap", "must be true")


def _validate_mesh(value: Any) -> None:
    data = _mapping(value, "mesh")
    for key in (
        "global_size_L0_m",
        "global_size_L1_m",
        "global_size_L2_m",
        "port_local_size_L0_m",
        "port_local_size_L1_m",
        "port_local_size_L2_m",
    ):
        _positive(data, key, "mesh")
    if _number(data, "global_size_L2_m", "mesh") > _number(data, "global_size_L1_m", "mesh"):
        _fail("mesh", "L2 must not be coarser than L1")
    if _number(data, "global_size_L1_m", "mesh") > _number(data, "global_size_L0_m", "mesh"):
        _fail("mesh", "L1 must not be coarser than L0")
    order = _number(data, "element_order", "mesh")
    if order < 1.0 or order != int(order):
        _fail("mesh.element_order", "must be a positive integer")
    _boolean(data, "pml_enabled", "mesh")
    if _number(data, "pml_element_count", "mesh") < 1.0:
        _fail("mesh.pml_element_count", "must be positive")


def _validate_study(value: Any) -> None:
    data = _mapping(value, "study")
    f_min = _positive(data, "frequency_min_Hz", "study")
    f_max = _positive(data, "frequency_max_Hz", "study")
    if f_max <= f_min:
        _fail("study", "frequency range is empty")
    frequency_count = _number(data, "frequency_count", "study")
    if frequency_count < 0.0:
        _fail("study.frequency_count", "must be nonnegative")
    if _number(data, "adaptive_peak_points", "study") < 1.0:
        _fail("study.adaptive_peak_points", "must be positive")
    fraction = _number(data, "reference_fraction", "study")
    if not 0.0 < fraction < 1.0:
        _fail("study.reference_fraction", "must be between zero and one")
    _string(data, "drive_mode", "study")
    if _boolean(data, "sweep_executed_in_phase1", "study"):
        _fail("study.sweep_executed_in_phase1", "phase 1 does not run sweeps")


def _validate_limits(value: Any) -> None:
    data = _mapping(value, "limits")
    _positive(data, "max_linear_displacement_m", "limits")
    _positive(data, "max_port_velocity_m_s", "limits")
    _positive(data, "max_pr_velocity_m_s", "limits")
    max_mach = _number(data, "max_mach", "limits")
    if not 0.0 < max_mach <= 1.0:
        _fail("limits.max_mach", "must be in (0, 1]")
    f_min = _positive(data, "trusted_frequency_min_Hz", "limits")
    f_max = _positive(data, "trusted_frequency_max_Hz", "limits")
    if f_max <= f_min:
        _fail("limits", "trusted frequency range is empty")


def _validate_topology(
    case: str,
    value: Any,
    port_enabled: bool,
    pr_enabled: bool,
    thermoviscous_enabled: bool,
) -> None:
    data = _mapping(value, "topology")
    expected = {
        "base_axisym": ("reference_only", False, False, False),
        "open_back": ("open_connected_free_field", False, False, False),
        "sealed_lossless": ("sealed_rigid", False, False, False),
        "sealed_thermoviscous": ("sealed_rigid", False, False, True),
        "vented_rear_coaxial": ("sealed_with_rear_coaxial_port", True, False, False),
        "passive_radiator_rear_coaxial": ("sealed_with_rear_coaxial_pr", False, True, False),
    }[case]
    if _string(data, "rear_boundary", "topology") != expected[0]:
        _fail("topology.rear_boundary", f"does not match case {case}")
    if _boolean(data, "port_enabled", "topology") != expected[1] or port_enabled != expected[1]:
        _fail("topology.port_enabled", "does not match case")
    if _boolean(data, "passive_radiator_enabled", "topology") != expected[2] or pr_enabled != expected[2]:
        _fail("topology.passive_radiator_enabled", "does not match case")
    if thermoviscous_enabled != expected[3]:
        _fail("topology", "thermoviscous choice does not match case")


def validate_enclosure_config(config: Mapping[str, Any]) -> EnclosureConfig:
    """Validate a phase-1 JSON-like mapping and return an immutable summary."""

    if not isinstance(config, Mapping):
        raise EnclosureSchemaError("config: must be an object")
    _walk_json(config)
    _check_unit_keys(config)
    case = config.get("case")
    if case not in ALLOWED_CASES:
        _fail("case", f"unknown case {case!r}")
    required_top = (
        "schema_version",
        "model_id",
        "title_cn",
        "demonstrator",
        "provenance",
        "physics",
        "geometry",
        "net_volume_target_m3",
        "volume_contract",
        "topology",
        "air",
        "port",
        "passive_radiator",
        "thermoviscous",
        "mesh",
        "study",
        "limits",
    )
    for key in required_top:
        _required(config, key, "config")
    if not _boolean(config, "demonstrator", "config"):
        _fail("demonstrator", "must be true for a non-product demonstrator")
    _string(config, "schema_version", "config")
    _string(config, "model_id", "config")
    title = _string(config, "title_cn", "config")
    if "演示" not in title or "非产品预测" not in title:
        _fail("title_cn", "must state 演示 and 非产品预测")
    _validate_provenance(config["provenance"])
    _validate_physics(config["physics"])
    geometry = _validate_geometry(config["geometry"])
    target = _positive(config, "net_volume_target_m3", "config")
    volume = _validate_volume(config["volume_contract"], target, geometry)
    port_decl = _mapping(config["port"], "port")
    pr_decl = _mapping(config["passive_radiator"], "passive_radiator")
    if port_decl.get("enabled") is True and pr_decl.get("enabled") is True:
        _fail("port/passive_radiator", "coaxial port and PR cannot occupy the same rear mount")
    port, port_enabled = _validate_port(port_decl)
    pr, pr_enabled = _validate_pr(pr_decl)
    _validate_thermoviscous(config["thermoviscous"])
    thermo_enabled = bool(config["thermoviscous"]["enabled"])
    _validate_topology(case, config["topology"], port_enabled, pr_enabled, thermo_enabled)
    if port_enabled and pr_enabled:
        _fail("port/passive_radiator", "coaxial port and PR cannot occupy the same rear mount")
    if port_enabled:
        if volume.port_displacement_m3 <= 0.0:
            _fail("volume_contract.port_displacement_m3", "enabled port needs positive occupancy")
        if volume.passive_radiator_back_displacement_m3 != 0.0 or volume.reserved_rear_feature_m3 != 0.0:
            _fail("volume_contract", "port case cannot also claim PR or rear reserve occupancy")
    elif pr_enabled:
        if volume.passive_radiator_back_displacement_m3 <= 0.0:
            _fail(
                "volume_contract.passive_radiator_back_displacement_m3",
                "enabled PR needs positive occupancy",
            )
        if volume.port_displacement_m3 != 0.0 or volume.reserved_rear_feature_m3 != 0.0:
            _fail("volume_contract", "PR case cannot also claim port or rear reserve occupancy")
    elif volume.port_displacement_m3 != 0.0 or volume.passive_radiator_back_displacement_m3 != 0.0:
        _fail("volume_contract", "disabled rear components cannot claim occupancy")

    if port_enabled:
        port_radius = _number(port, "radius_m", "port")
        opening = geometry["rear_opening_radius_m"]
        if not geometry["rear_opening_enabled"] or opening is None or port_radius > float(opening):
            _fail("port.radius_m", "port does not fit the rear opening")
        if port_radius + float(geometry["driver_radius_m"]) > float(geometry["inner_radius_m"]):
            _fail("port", "port and driver geometries intersect in the coaxial cavity")
        if float(geometry["port_penetration_into_box_m"]) + float(
            geometry["driver_projection_length_m"]
        ) >= float(geometry["inner_depth_m"]):
            _fail("port", "port and driver geometries intersect")
    if pr_enabled:
        sd = _number(pr, "Sd_m2", "passive_radiator")
        rear_area = math.pi * float(geometry["rear_mount_radius_m"]) ** 2
        if sd > rear_area:
            _fail("passive_radiator.Sd_m2", "PR area exceeds the rear wall")
        opening = geometry["rear_opening_radius_m"]
        pr_radius = math.sqrt(sd / math.pi)
        if not geometry["rear_opening_enabled"] or opening is None or pr_radius > float(opening):
            _fail("passive_radiator.Sd_m2", "PR area does not fit the rear opening")

    # Treat the PML as a spherical radial boundary around the demonstrator.
    exterior_axial_extent = float(geometry["outer_depth_m"]) / 2.0
    if port_enabled:
        exterior_axial_extent += _number(port, "length_m", "port")
    if pr_enabled:
        exterior_axial_extent += _number(pr, "rear_clearance_m", "passive_radiator")
    entity_radius = math.hypot(float(geometry["outer_radius_m"]), exterior_axial_extent)
    if float(geometry["pml_inner_radius_m"]) <= entity_radius:
        _fail("geometry.pml_inner_radius_m", "PML inner boundary crosses an entity")

    _validate_air(config["air"])
    _validate_mesh(config["mesh"])
    _validate_study(config["study"])
    _validate_limits(config["limits"])
    return EnclosureConfig(
        raw=deepcopy(dict(config)),
        case=case,
        demonstrator=True,
        net_volume_target_m3=target,
        computed_net_volume_m3=volume.computed_net_volume_m3,
        volume_contract=volume,
    )


def _reject_json_constant(token: str) -> None:
    raise EnclosureSchemaError(f"JSON numeric constant {token} is not finite")


def load_enclosure_config(path: str | Path) -> EnclosureConfig:
    """Read JSON with non-standard NaN/Infinity constants rejected."""

    config_path = Path(path)
    try:
        data = json.loads(
            config_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except EnclosureSchemaError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise EnclosureSchemaError(f"{config_path}: cannot read valid JSON: {exc}") from exc
    return validate_enclosure_config(data)


# Friendly aliases for callers that do not need the longer function names.
validate_config = validate_enclosure_config
load_config = load_enclosure_config
