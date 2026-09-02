"""Reference-only A/B validation and temporary scan helpers for Stage 3B2.

This module deliberately sits above :mod:`enclosure_acoustics`.  It generates
only temporary reference meshes, records deterministic scalar diagnostics, and
never writes the repository's formal configuration, ``runs`` directory, or
production input mesh.  The returned records distinguish the actual 1 m
near-field reconstruction from the closed-HK far-field normalization.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .enclosure_acoustics import (
    EXPLICIT_PML_MODE,
    REFERENCE_PLANAR_PISTON_BACK,
    REFERENCE_PLANAR_PISTON_FRONT,
    ReferencePrescribedVelocityAcoustics,
    sha256_file,
)
from .enclosure_geometry import case_id_for_config, generate_reference_mesh
from .enclosure_schema import load_enclosure_config


VALID_CASES = ("A", "B")
VALID_LEVELS = ("L0", "L1", "L2")
DEFAULT_REFERENCE_ALPHA = 4.0
GRID_AMPLITUDE_LIMIT_DB = 0.3
GRID_PHASE_LIMIT_DEG = 3.0
POWER_BALANCE_LIMIT = 0.02
VOLUME_VELOCITY_LIMIT = 0.005


def _as_float_sequence(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _as_case_config_map(config_paths: Mapping[str, str | Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw_case, raw_path in config_paths.items():
        case = str(raw_case).upper()
        if case not in VALID_CASES:
            raise ValueError(f"reference validation supports A/B only, got {raw_case!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        inferred = case_id_for_config(load_enclosure_config(path).case)
        if inferred != case:
            raise ValueError(f"config {path} maps to {inferred}, not requested {case}")
        result[case] = path
    if not result:
        raise ValueError("at least one A/B config is required")
    return {case: result[case] for case in VALID_CASES if case in result}


def _complex_from_json(value: Mapping[str, Any]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def phase_difference_deg(first: complex | float, second: complex | float) -> float:
    """Return the wrapped phase difference ``first-second`` in degrees."""

    if not isinstance(first, complex):
        first = complex(first)
    if not isinstance(second, complex):
        second = complex(second)
    return float(np.angle(first / second, deg=True))


def amplitude_difference_db(first: complex | float, second: complex | float) -> float:
    """Return ``20 log10(abs(first)/abs(second))`` in dB."""

    first_abs = max(abs(first), 1.0e-300)
    second_abs = max(abs(second), 1.0e-300)
    return float(20.0 * math.log10(first_abs / second_abs))


def fit_db_per_decade(
    frequencies_Hz: Sequence[float],
    amplitude_ratios: Sequence[float],
) -> float:
    """Fit amplitude-ratio dB against log10 frequency independently."""

    frequencies = np.asarray(frequencies_Hz, dtype=float)
    ratios = np.asarray(amplitude_ratios, dtype=float)
    if frequencies.ndim != 1 or ratios.ndim != 1 or len(frequencies) != len(ratios):
        raise ValueError("frequencies and ratios must be one-dimensional and equal length")
    if len(frequencies) < 2 or np.any(frequencies <= 0.0) or np.any(ratios <= 0.0):
        raise ValueError("slope fit needs at least two positive frequency/ratio pairs")
    coefficients = np.polyfit(np.log10(frequencies), 20.0 * np.log10(ratios), 1)
    return float(coefficients[0])


def make_temporary_geometry_config(
    source_path: str | Path,
    destination_dir: str | Path,
    *,
    variant: str = "base",
    pml_inner_radius_m: float | None = None,
    pml_thickness_m: float | None = None,
) -> Path:
    """Copy one formal A/B config and apply only temporary PML geometry edits."""

    source = Path(source_path)
    source_config = load_enclosure_config(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    geometry = raw["geometry"]
    if pml_inner_radius_m is not None:
        geometry["pml_inner_radius_m"] = float(pml_inner_radius_m)
    if pml_thickness_m is not None:
        geometry["pml_thickness_m"] = float(pml_thickness_m)
    if float(geometry["pml_inner_radius_m"]) + float(geometry["pml_thickness_m"]) >= 1.0:
        raise ValueError("temporary geometry must keep the 1 m evaluation point outside HK/PML")

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    case = case_id_for_config(source_config.case)
    safe_variant = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(variant))
    target = destination / f"{case}_{safe_variant}.json"
    target.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied = load_enclosure_config(target)
    if float(copied.raw["net_volume_target_m3"]) != float(source_config.raw["net_volume_target_m3"]):
        raise ValueError("temporary geometry edit changed net_volume_target_m3")
    return target


def _mesh_paths(
    config_paths: Mapping[str, Path],
    levels: Sequence[str],
    mesh_dir: str | Path,
) -> tuple[dict[tuple[str, str], Path], dict[str, dict[str, Any]]]:
    directory = Path(mesh_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, str], Path] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for case in (case for case in VALID_CASES if case in config_paths):
        config = config_paths[case]
        config_hash = sha256_file(config)
        for level in levels:
            path = directory / f"{case}_{level}_{config_hash[:12]}.msh"
            if not path.exists():
                generate_reference_mesh(config, level, path)
            paths[(case, level)] = path
            manifest[f"{case}/{level}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "config_path": str(config),
                "config_sha256": config_hash,
                "level": level,
                "case": case,
            }
    return paths, manifest


def _solve_row(
    model: ReferencePrescribedVelocityAcoustics,
    mesh_path: Path,
    config_path: Path,
    case: str,
    level: str,
    frequency_Hz: float,
    pml_alpha: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = model.solve(frequency_Hz)
    solve_wall_s = time.perf_counter() - started
    payload = result.as_dict()
    hk = result.hk_diagnostics
    actual = hk["actual_pressure_1m"]
    far = hk["far_field_normalized_to_1m"]
    front = result.front_back_traces[REFERENCE_PLANAR_PISTON_FRONT]
    back = result.front_back_traces[REFERENCE_PLANAR_PISTON_BACK]
    driver_radius = float(model.mesh_data.config.raw["geometry"]["driver_radius_m"])
    reference_volume_velocity = math.pi * driver_radius * driver_radius * float(
        model.parameters.reference_velocity_m_s
    )
    q_error = max(
        abs(abs(float(front["q_out_m3_s"])) - reference_volume_velocity),
        abs(abs(float(back["q_out_m3_s"])) - reference_volume_velocity),
    ) / max(reference_volume_velocity, 1.0e-30)
    return {
        "case": case,
        "level": level,
        "frequency_Hz": float(frequency_Hz),
        "pml_mode": EXPLICIT_PML_MODE,
        "pml_alpha": float(pml_alpha),
        "status": "pass",
        "reference_identity": "reference planar piston",
        "final_production_interface_ready": False,
        "mesh_path": str(mesh_path),
        "mesh_sha256": model.mesh_data.source_sha256,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pressure_dof_count": int(result.pressure.size),
        "pressure_triangle_count": int(model.mesh_data.pressure_triangle_count),
        "cavity_volume_m3": float(model.mesh_data.cavity_volume_m3),
        "actual_pressure_1m": actual,
        "far_field_normalized_to_1m": far,
        "q_front_out_m3_s": float(front["q_out_m3_s"]),
        "q_back_out_m3_s": float(back["q_out_m3_s"]),
        "q_out_total_m3_s": float(result.q_out_total_m3_s),
        "reference_volume_velocity_m3_s": float(reference_volume_velocity),
        "volume_velocity_error_relative": float(q_error),
        "pin_W": float(result.drive_power_into_fluid_W["total"]),
        "phk_W": float(hk["hk_flux_power_W"]),
        "pml_discrete_absorption_W": float(
            result.pml_diagnostics["discrete_absorption_power_W"]
        ),
        "power_balance_relative": float(hk["power_balance_relative_to_drive"]),
        "input_power_boundary_rhs_error_W": float(
            result.input_power_boundary_cross_error_W
        ),
        "linear_residual_relative": float(result.residual_relative),
        "pml_passivity_status": result.pml_diagnostics["passivity_status"],
        "audit_status": model.mesh_data.audit_report["status"],
        "solve_wall_s": float(solve_wall_s),
    }


def _reference_rows(rows: Sequence[Mapping[str, Any]], alpha: float) -> list[Mapping[str, Any]]:
    return [row for row in rows if math.isclose(float(row["pml_alpha"]), alpha)]


def _convergence_rows(rows: Sequence[Mapping[str, Any]], alpha: float) -> list[dict[str, Any]]:
    selected = _reference_rows(rows, alpha)
    index = {(row["case"], row["level"], row["frequency_Hz"]): row for row in selected}
    result: list[dict[str, Any]] = []
    for case in sorted({str(row["case"]) for row in selected}, key=VALID_CASES.index):
        frequencies = sorted(
            frequency
            for (row_case, level, frequency) in index
            if row_case == case and level in {"L1", "L2"}
        )
        for frequency in sorted(set(frequencies)):
            coarse = index.get((case, "L1", frequency))
            fine = index.get((case, "L2", frequency))
            if coarse is None or fine is None:
                continue
            coarse_pressure = _complex_from_json(
                coarse["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"]
            )
            fine_pressure = _complex_from_json(
                fine["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"]
            )
            coarse_actual = _complex_from_json(coarse["actual_pressure_1m"]["pressure_Pa"])
            fine_actual = _complex_from_json(fine["actual_pressure_1m"]["pressure_Pa"])
            result.append(
                {
                    "case": case,
                    "frequency_Hz": float(frequency),
                    "reference_alpha": float(alpha),
                    "far_field_amplitude_change_dB": amplitude_difference_db(
                        fine_pressure, coarse_pressure
                    ),
                    "far_field_phase_change_deg": phase_difference_deg(
                        fine_pressure, coarse_pressure
                    ),
                    "actual_1m_amplitude_change_dB": amplitude_difference_db(
                        fine_actual, coarse_actual
                    ),
                    "actual_1m_phase_change_deg": phase_difference_deg(
                        fine_actual, coarse_actual
                    ),
                    "l2_power_balance_relative": float(fine["power_balance_relative"]),
                    "l2_volume_velocity_error_relative": float(
                        fine["volume_velocity_error_relative"]
                    ),
                    "passes_far_field_grid_gate": bool(
                        abs(amplitude_difference_db(fine_pressure, coarse_pressure))
                        < GRID_AMPLITUDE_LIMIT_DB
                        and abs(phase_difference_deg(fine_pressure, coarse_pressure))
                        < GRID_PHASE_LIMIT_DEG
                    ),
                }
            )
    return result


def _case_comparison(rows: Sequence[Mapping[str, Any]], alpha: float) -> dict[str, Any]:
    selected = _reference_rows(rows, alpha)
    if not {str(row["case"]) for row in selected}.issuperset({"A", "B"}):
        return {}
    index = {(row["case"], row["level"], row["frequency_Hz"]): row for row in selected}
    levels = sorted({row["level"] for row in selected}, key=VALID_LEVELS.index)
    comparison: dict[str, Any] = {}
    for level in levels:
        points: list[dict[str, Any]] = []
        for frequency in sorted(
            {
                row["frequency_Hz"]
                for row in selected
                if row["level"] == level
            }
        ):
            a = index.get(("A", level, frequency))
            b = index.get(("B", level, frequency))
            if a is None or b is None:
                continue
            a_far = _complex_from_json(a["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"])
            b_far = _complex_from_json(b["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"])
            a_actual = _complex_from_json(a["actual_pressure_1m"]["pressure_Pa"])
            b_actual = _complex_from_json(b["actual_pressure_1m"]["pressure_Pa"])
            points.append(
                {
                    "frequency_Hz": float(frequency),
                    "far_field_ratio_A_over_B": float(abs(a_far) / max(abs(b_far), 1.0e-300)),
                    "far_field_difference_dB": amplitude_difference_db(a_far, b_far),
                    "far_field_phase_difference_deg": phase_difference_deg(a_far, b_far),
                    "actual_1m_ratio_A_over_B": float(
                        abs(a_actual) / max(abs(b_actual), 1.0e-300)
                    ),
                    "actual_1m_difference_dB": amplitude_difference_db(a_actual, b_actual),
                    "actual_1m_phase_difference_deg": phase_difference_deg(
                        a_actual, b_actual
                    ),
                }
            )
        entry: dict[str, Any] = {"points": points}
        if len(points) >= 2:
            entry["far_field_slope_dB_per_decade"] = fit_db_per_decade(
                [point["frequency_Hz"] for point in points],
                [point["far_field_ratio_A_over_B"] for point in points],
            )
            entry["actual_1m_slope_dB_per_decade"] = fit_db_per_decade(
                [point["frequency_Hz"] for point in points],
                [point["actual_1m_ratio_A_over_B"] for point in points],
            )
        comparison[level] = entry
    return comparison


def _alpha_comparison(rows: Sequence[Mapping[str, Any]], reference_alpha: float) -> list[dict[str, Any]]:
    index = {
        (row["case"], row["level"], row["frequency_Hz"], row["pml_alpha"]): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for key, reference in sorted(index.items(), key=lambda item: item[0]):
        case, level, frequency, alpha = key
        if case != "A" or level != "L1" or math.isclose(float(alpha), reference_alpha):
            continue
        baseline = index.get((case, level, frequency, reference_alpha))
        if baseline is None:
            continue
        pressure = _complex_from_json(reference["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"])
        baseline_pressure = _complex_from_json(
            baseline["far_field_normalized_to_1m"]["pressure_ff_1m_Pa"]
        )
        result.append(
            {
                "case": case,
                "level": level,
                "frequency_Hz": float(frequency),
                "alpha": float(alpha),
                "relative_to_alpha4_dB": amplitude_difference_db(pressure, baseline_pressure),
                "relative_to_alpha4_phase_deg": phase_difference_deg(pressure, baseline_pressure),
                "passive": reference["pml_passivity_status"] == "pass",
            }
        )
    return result


def scan_reference_cases(
    config_paths: Mapping[str, str | Path],
    levels: Sequence[str],
    frequencies_Hz: Sequence[float],
    *,
    mesh_dir: str | Path,
    pml_alphas: Sequence[float] = (DEFAULT_REFERENCE_ALPHA,),
) -> dict[str, Any]:
    """Generate and solve a deterministic temporary A/B reference scan."""

    configs = _as_case_config_map(config_paths)
    normalized_levels = tuple(sorted({str(level).upper() for level in levels}, key=VALID_LEVELS.index))
    if not normalized_levels or any(level not in VALID_LEVELS for level in normalized_levels):
        raise ValueError("levels must be selected from L0, L1, L2")
    frequencies = _as_float_sequence(frequencies_Hz, "frequencies_Hz")
    alphas = _as_float_sequence(pml_alphas, "pml_alphas")
    if any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("explicit PML alpha values must be positive")
    reference_alpha = 4.0 if any(math.isclose(alpha, 4.0) for alpha in alphas) else alphas[0]
    mesh_paths, mesh_manifest = _mesh_paths(configs, normalized_levels, mesh_dir)
    models = {
        (case, level, alpha): ReferencePrescribedVelocityAcoustics.from_files(
            mesh_paths[(case, level)],
            configs[case],
            pml_mode=EXPLICIT_PML_MODE,
            pml_alpha=alpha,
        )
        for case in configs
        for level in normalized_levels
        for alpha in alphas
    }
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for case in configs:
        for level in normalized_levels:
            for alpha in alphas:
                model = models[(case, level, alpha)]
                for frequency in frequencies:
                    rows.append(
                        _solve_row(
                            model,
                            mesh_paths[(case, level)],
                            configs[case],
                            case,
                            level,
                            frequency,
                            alpha,
                        )
                    )
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: (row["case"], VALID_LEVELS.index(row["level"]), row["pml_alpha"], row["frequency_Hz"]))
    convergence = _convergence_rows(rows, reference_alpha)
    comparison = _case_comparison(rows, reference_alpha)
    far_points = comparison.get("L1", {}).get("points", [])
    far_slope = comparison.get("L1", {}).get("far_field_slope_dB_per_decade")
    report = {
        "schema": "luna.enclosure_stage3b2_validation.v1",
        "title": "Stage 3B2 reference A/B validation scan",
        "scope": "reference planar piston A/B only; no production, thermal, port, or PR coupling",
        "identity": {
            "source": "reference planar piston",
            "prescribed_velocity": "global v_z = +1 m/s",
            "time_convention": "exp(+i omega t)",
            "final_production_interface_ready": False,
        },
        "cases": list(configs),
        "levels": list(normalized_levels),
        "frequencies_Hz": list(frequencies),
        "pml_mode": EXPLICIT_PML_MODE,
        "reference_alpha": float(reference_alpha),
        "pml_alphas": list(alphas),
        "configs": {
            case: {
                "path": str(configs[case]),
                "sha256": sha256_file(configs[case]),
            }
            for case in configs
        },
        "mesh_manifest": mesh_manifest,
        "rows": rows,
        "grid_convergence": convergence,
        "case_comparison": comparison,
        "alpha_comparison": _alpha_comparison(rows, reference_alpha),
        "acceptance": {
            "far_field_dipole_slope_dB_per_decade": far_slope if far_slope is not None else "not_evaluated",
            "far_field_dipole_points": far_points,
            "far_field_dipole_gate": (
                bool(18.0 <= far_slope <= 22.0) if far_slope is not None else "not_evaluated"
            ),
            "l1_to_l2_amplitude_limit_dB": GRID_AMPLITUDE_LIMIT_DB,
            "l1_to_l2_phase_limit_deg": GRID_PHASE_LIMIT_DEG,
            "l2_power_balance_limit_relative": POWER_BALANCE_LIMIT,
            "volume_velocity_limit_relative": VOLUME_VELOCITY_LIMIT,
            "l0_is_final_conclusion": False,
            "grid_gate": bool(
                convergence
                and all(item["passes_far_field_grid_gate"] for item in convergence)
            ),
        },
        "timing": {
            "solve_count": len(rows),
            "wall_time_s": float(elapsed),
        },
        "power_interpretation": (
            "PHK and discrete PML absorption are numerical reference-domain flux "
            "diagnostics; PML absorption is not material dissipation."
        ),
    }
    return report


CSV_FIELDS = (
    "case",
    "level",
    "frequency_Hz",
    "pml_alpha",
    "status",
    "pressure_dof_count",
    "pressure_triangle_count",
    "mesh_sha256",
    "actual_1m_rms_spl_dB",
    "actual_1m_phase_deg",
    "actual_1m_field_regime",
    "far_field_R_eval_m",
    "far_field_kR_eval",
    "far_field_rms_spl_dB",
    "far_field_phase_deg",
    "far_field_pressure_real_Pa",
    "far_field_pressure_imag_Pa",
    "pin_W",
    "phk_W",
    "pml_discrete_absorption_W",
    "power_balance_relative",
    "volume_velocity_error_relative",
    "solve_wall_s",
)


def write_validation_outputs(
    report: Mapping[str, Any],
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write stable JSON and an intentionally compact scalar CSV."""

    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if csv_path is None:
        return
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            actual = row["actual_pressure_1m"]
            far = row["far_field_normalized_to_1m"]
            pressure = far["pressure_ff_1m_Pa"]
            writer.writerow(
                {
                    **row,
                    "actual_1m_rms_spl_dB": actual["rms_spl_dB"],
                    "actual_1m_phase_deg": actual["phase_deg"],
                    "actual_1m_field_regime": actual["field_regime"],
                    "far_field_R_eval_m": far["R_eval_m"],
                    "far_field_kR_eval": far["kR_eval"],
                    "far_field_rms_spl_dB": far["rms_spl_dB"],
                    "far_field_phase_deg": far["phase_deg"],
                    "far_field_pressure_real_Pa": pressure["real"],
                    "far_field_pressure_imag_Pa": pressure["imag"],
                }
            )


__all__ = [
    "DEFAULT_REFERENCE_ALPHA",
    "GRID_AMPLITUDE_LIMIT_DB",
    "GRID_PHASE_LIMIT_DEG",
    "POWER_BALANCE_LIMIT",
    "VALID_CASES",
    "VALID_LEVELS",
    "VOLUME_VELOCITY_LIMIT",
    "amplitude_difference_db",
    "fit_db_per_decade",
    "make_temporary_geometry_config",
    "phase_difference_deg",
    "scan_reference_cases",
    "write_validation_outputs",
]
