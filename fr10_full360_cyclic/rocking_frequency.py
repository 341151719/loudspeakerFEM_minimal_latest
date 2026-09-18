"""Frequency-domain rocking-mode analysis on the FR10 cyclic 3-D FEM.

The symmetric ``k=0`` solve supplies coil displacement and current.  A second
``k=1`` solve uses a normalized moment load to obtain the radiation-loaded
rocking compliance.  The three weak-asymmetry source moments then follow the
feed-forward separation in Cardenas and Klippel:

    mu_mass      = omega**2 * Delta_m * X
    mu_stiffness = -Delta_k * X
    mu_Bl        = Delta_Bl_moment * I

This is intentionally a first-order root-cause model.  The defects do not feed
back into the piston/electrical solution and do not couple the four Bloch
classes inside a single matrix solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import base_p2_local_solver as base
import cyclic_full360_solver as cyclic


HERE = Path(__file__).resolve().parent


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _phasor(magnitude: float, phase_deg: float = 0.0) -> complex:
    return float(magnitude) * np.exp(1j * math.radians(float(phase_deg)))


def load_rocking_config(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else HERE / "configs/rocking_modes_frequency.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _full360_resultant_moment(
    model: dict[str, Any], sector_force: np.ndarray, kclass: int = 1
) -> np.ndarray:
    """Return the complex full-360 moment of a cyclic sector nodal load."""
    points = np.asarray(model["P"], float)
    force = np.asarray(sector_force, complex).reshape(-1, 3)
    if force.shape != points.shape:
        raise ValueError("sector force and structural points do not match")
    moment = np.zeros(3, complex)
    phase = np.exp(1j * int(kclass) * math.pi / 2.0)
    for quadrant in range(4):
        angle = quadrant * math.pi / 2.0
        ca, sa = math.cos(angle), math.sin(angle)
        rotation = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        xyz = points @ rotation.T
        load = (phase**quadrant) * (force @ rotation.T)
        moment += np.cross(xyz, load).sum(axis=0)
    return moment


def unit_rocking_moment_force(
    model: dict[str, Any], kclass: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Create a coil-distributed cyclic load with unit transverse moment.

    The existing volume-average coil force is continued with the requested
    Bloch phase.  Its zero-order resultant cancels for k=1, while its transverse
    resultant moment is normalized to one N m.
    """
    force = np.asarray(model["gcoil"], complex).copy()
    moment = _full360_resultant_moment(model, force, kclass=kclass)
    transverse_norm = float(np.linalg.norm(moment[:2]))
    if transverse_norm <= 1e-15:
        raise RuntimeError("cyclic coil load has zero transverse moment")
    force /= transverse_norm
    normalized = _full360_resultant_moment(model, force, kclass=kclass)
    return force, normalized


def extract_coil_tilt(
    model: dict[str, Any], sector_displacement: np.ndarray, kclass: int = 1
) -> dict[str, complex | float]:
    """Fit u_z = offset + duz_dx*x + duz_dy*y on the full-360 coil."""
    points = np.asarray(model["P"], float)
    displacement = np.asarray(sector_displacement, complex).reshape(-1, 3)
    weights = np.abs(np.asarray(model["gcoil"])[2::3])
    selected = weights > max(float(weights.max(initial=0.0)) * 1e-12, 1e-30)
    phase = np.exp(1j * int(kclass) * math.pi / 2.0)
    design: list[np.ndarray] = []
    values: list[np.ndarray] = []
    fit_weights: list[np.ndarray] = []
    for quadrant in range(4):
        angle = quadrant * math.pi / 2.0
        ca, sa = math.cos(angle), math.sin(angle)
        rotation = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        xyz = points @ rotation.T
        field = (phase**quadrant) * (displacement @ rotation.T)
        design.append(
            np.column_stack(
                (np.ones(np.count_nonzero(selected)), xyz[selected, 0], xyz[selected, 1])
            )
        )
        values.append(field[selected, 2])
        fit_weights.append(weights[selected])
    matrix = np.vstack(design)
    rhs = np.concatenate(values)
    weight = np.sqrt(np.concatenate(fit_weights))
    coeff, *_ = np.linalg.lstsq(matrix * weight[:, None], rhs * weight, rcond=None)
    fitted = matrix @ coeff
    residual = float(
        np.linalg.norm(weight * (rhs - fitted))
        / max(np.linalg.norm(weight * rhs), 1e-300)
    )
    # Small-rotation kinematics: u_z = theta_x*y - theta_y*x.
    tilt_x = complex(coeff[2])
    tilt_y = complex(-coeff[1])
    return {
        "offset_m": complex(coeff[0]),
        "duz_dx": complex(coeff[1]),
        "duz_dy": complex(coeff[2]),
        "tilt_x_rad": tilt_x,
        "tilt_y_rad": tilt_y,
        "tilt_magnitude_rad": float(math.sqrt(abs(tilt_x) ** 2 + abs(tilt_y) ** 2)),
        "fit_relative_residual": residual,
    }


def root_cause_moments(
    frequency_Hz: float,
    coil_displacement_m: complex,
    current_A: complex,
    root_cfg: dict[str, Any],
) -> dict[str, complex]:
    """Evaluate the separated mass, stiffness and force-factor moments."""
    omega = 2.0 * math.pi * float(frequency_Hz)
    mass = _phasor(
        root_cfg.get("mass_unbalance_kg_m", 0.0),
        root_cfg.get("mass_direction_deg", 0.0),
    )
    stiffness = _phasor(
        root_cfg.get("stiffness_coupling_N", 0.0),
        root_cfg.get("stiffness_direction_deg", 0.0),
    )
    bl_moment = _phasor(
        root_cfg.get("bl_moment_Nm_per_A", 0.0),
        root_cfg.get("bl_direction_deg", 0.0),
    )
    result = {
        "mass_Nm": omega**2 * mass * complex(coil_displacement_m),
        "stiffness_Nm": -stiffness * complex(coil_displacement_m),
        "bl_Nm": bl_moment * complex(current_A),
    }
    result["total_Nm"] = sum(result.values(), 0.0j)
    return result


def _projected_rocking_compliance(
    applied_moment: np.ndarray, tilt: dict[str, complex | float]
) -> complex:
    moment = np.asarray(applied_moment[:2], complex)
    response = np.array([tilt["tilt_x_rad"], tilt["tilt_y_rad"]], complex)
    denominator = np.vdot(moment, moment)
    if abs(denominator) <= 1e-30:
        raise RuntimeError("cannot project rocking compliance onto a zero moment")
    return complex(np.vdot(moment, response) / denominator)


def analyze_frequency(
    frequency_Hz: float,
    cfg: dict[str, Any],
    rocking_cfg: dict[str, Any],
    model: dict[str, Any],
    front: dict[str, Any],
    rear: dict[str, Any],
    coupling,
    suspension_scale: float,
) -> dict[str, Any]:
    frequency = float(frequency_Hz)
    voltage = float(rocking_cfg.get("drive_voltage_peak_V", cfg.get("drive_voltage_peak_V", 1.0)))
    piston_u, _, _, piston_meta = cyclic.solve_phase(
        cfg, model, front, rear, coupling, frequency, suspension_scale, 0
    )
    omega = 2.0 * math.pi * frequency
    bl = float(cfg["electrical"]["Bl_Tm"])
    displacement_per_A = complex(model["gcoil"] @ piston_u)
    motional_impedance = 1j * omega * bl * displacement_per_A
    blocked_impedance = complex(cfg["electrical"]["Rdc_ohm"], omega * cfg["electrical"]["Le_H"])
    total_impedance = blocked_impedance + motional_impedance
    current = voltage / total_impedance
    coil_displacement = displacement_per_A * current

    moment_force, applied_moment = unit_rocking_moment_force(model, kclass=1)
    rocking_u, _, _, rocking_meta = cyclic.solve_phase(
        cfg,
        model,
        front,
        rear,
        coupling,
        frequency,
        suspension_scale,
        1,
        force_full=moment_force,
    )
    tilt = extract_coil_tilt(model, rocking_u, kclass=1)
    compliance = _projected_rocking_compliance(applied_moment, tilt)
    moments = root_cause_moments(
        frequency, coil_displacement, current, rocking_cfg.get("root_causes", {})
    )
    tilts = {name.replace("_Nm", "_rad"): compliance * value for name, value in moments.items()}
    backward_tolerance = float(cfg["numerics"]["backward_error_tolerance"])
    residual_tolerance = float(cfg["numerics"]["relative_residual_tolerance"])
    acceptance = {
        "unit_moment_relative_error": float(abs(np.linalg.norm(applied_moment[:2]) - 1.0)),
        "tilt_fit_relative_residual": float(tilt["fit_relative_residual"]),
        "piston_backward_error_pass": bool(piston_meta["normwise_backward_error_inf"] < backward_tolerance),
        "rocking_backward_error_pass": bool(rocking_meta["normwise_backward_error_inf"] < backward_tolerance),
        "piston_residual_pass": bool(piston_meta["relative_residual_2"] < residual_tolerance),
        "rocking_residual_pass": bool(rocking_meta["relative_residual_2"] < residual_tolerance),
    }
    acceptance["pass"] = bool(
        acceptance["unit_moment_relative_error"] < 1e-10
        and acceptance["tilt_fit_relative_residual"] < 0.05
        and all(value for key, value in acceptance.items() if key.endswith("_pass"))
    )

    return {
        "frequency_Hz": frequency,
        "formulation": "weak-asymmetry feed-forward root-cause separation using k=0 piston and k=1 radiation-loaded 3-D FEM compliance",
        "rocking_basis": "k=1 circular member of the m=1 doublet; a fixed-axis real rocking motion is recovered by combining the conjugate k=1/k=3 pair",
        "electrical": {
            "voltage_V_peak": voltage,
            "current_A_peak": _complex_pair(current),
            "blocked_impedance_ohm": _complex_pair(blocked_impedance),
            "motional_impedance_ohm": _complex_pair(motional_impedance),
            "total_impedance_ohm": _complex_pair(total_impedance),
        },
        "piston": {
            "coil_displacement_m_peak": _complex_pair(coil_displacement),
            "displacement_per_A_m_A": _complex_pair(displacement_per_A),
        },
        "rocking_resonator": {
            "unit_moment_vector_Nm": [_complex_pair(v) for v in applied_moment],
            "compliance_rad_per_Nm": _complex_pair(compliance),
            "compliance_magnitude_rad_per_Nm": float(abs(compliance)),
            "compliance_phase_deg": float(np.degrees(np.angle(compliance))),
            "unit_load_tilt": {
                key: (_complex_pair(value) if isinstance(value, complex) else value)
                for key, value in tilt.items()
            },
        },
        "root_cause_moments": {key: _complex_pair(value) for key, value in moments.items()},
        "root_cause_tilts": {
            key: {
                "complex_rad_peak": _complex_pair(value),
                "magnitude_rad_peak": float(abs(value)),
                "phase_deg": float(np.degrees(np.angle(value))),
            }
            for key, value in tilts.items()
        },
        "acceptance": acceptance,
        "solver": {"piston_k0": piston_meta, "rocking_k1": rocking_meta},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    causes = ("mass", "stiffness", "bl", "total")
    fields = [
        "frequency_Hz",
        "current_magnitude_A_peak",
        "coil_displacement_magnitude_m_peak",
        "rocking_compliance_magnitude_rad_per_Nm",
        "rocking_compliance_phase_deg",
    ]
    for cause in causes:
        fields += [f"moment_{cause}_magnitude_Nm", f"tilt_{cause}_magnitude_rad_peak", f"tilt_{cause}_phase_deg"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            current = complex(*row["electrical"]["current_A_peak"])
            displacement = complex(*row["piston"]["coil_displacement_m_peak"])
            flat: dict[str, float] = {
                "frequency_Hz": row["frequency_Hz"],
                "current_magnitude_A_peak": abs(current),
                "coil_displacement_magnitude_m_peak": abs(displacement),
                "rocking_compliance_magnitude_rad_per_Nm": row["rocking_resonator"]["compliance_magnitude_rad_per_Nm"],
                "rocking_compliance_phase_deg": row["rocking_resonator"]["compliance_phase_deg"],
            }
            for cause in causes:
                moment = complex(*row["root_cause_moments"][f"{cause}_Nm"])
                tilt = row["root_cause_tilts"][f"{cause}_rad"]
                flat[f"moment_{cause}_magnitude_Nm"] = abs(moment)
                flat[f"tilt_{cause}_magnitude_rad_peak"] = tilt["magnitude_rad_peak"]
                flat[f"tilt_{cause}_phase_deg"] = tilt["phase_deg"]
            writer.writerow(flat)


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = np.array([row["frequency_Hz"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for cause, label in (("mass", "mass"), ("stiffness", "stiffness"), ("bl", "Bl"), ("total", "total")):
        magnitude = np.array(
            [row["root_cause_tilts"][f"{cause}_rad"]["magnitude_rad_peak"] for row in rows]
        )
        axes[0].loglog(frequency, np.maximum(magnitude, 1e-30), label=label)
    axes[0].set_ylabel("tilt / rad peak")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[0].set_title("FR10 frequency-domain rocking root-cause separation")
    compliance = np.array(
        [row["rocking_resonator"]["compliance_magnitude_rad_per_Nm"] for row in rows]
    )
    axes[1].loglog(frequency, np.maximum(compliance, 1e-30), color="black")
    axes[1].set_xlabel("frequency / Hz")
    axes[1].set_ylabel("|H rocking| / rad per N m")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_rocking_sweep(
    frequencies: list[float] | tuple[float, ...] | None = None,
    config_path: str | Path | None = None,
    output: str | Path | None = None,
    suspension_scale: float | None = None,
) -> dict[str, Any]:
    cfg = cyclic.load_cfg()
    rocking_cfg = load_rocking_config(config_path)
    selected = list(frequencies or rocking_cfg["frequencies_Hz"])
    if not selected or any(float(value) <= 0.0 for value in selected):
        raise ValueError("frequencies must be positive")
    scale = float(suspension_scale or cfg["calibration"]["p2_local_asb_suspension_scale"])
    out = Path(output or cyclic.default_output_root() / "rocking_frequency")
    out.mkdir(parents=True, exist_ok=True)
    model = cyclic.build_sector_model(cfg)
    front, rear = base.build_acoustic_domains(cfg)
    coupling, coupling_report = base.build_local_G(model, front, cfg)
    rows = [
        analyze_frequency(value, cfg, rocking_cfg, model, front, rear, coupling, scale)
        for value in selected
    ]
    summary = {
        "status": "pass" if all(row["acceptance"]["pass"] for row in rows) else "fail",
        "physics_scope": "linear frequency-domain weak-asymmetry feed-forward model; no k0-k1 feedback and no 3-D MQS",
        "config": rocking_cfg,
        "suspension_scale": scale,
        "local_asb": coupling_report,
        "frequencies": rows,
    }
    (out / "rocking_frequency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(out / "rocking_frequency_response.csv", rows)
    _write_plot(out / "rocking_frequency_response.png", rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FR10 frequency-domain rocking-mode analysis")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freq", type=float, nargs="+")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_rocking_sweep(args.freq, args.config, args.out, args.scale)
    print(json.dumps({"status": result["status"], "frequencies": len(result["frequencies"])}, indent=2))
