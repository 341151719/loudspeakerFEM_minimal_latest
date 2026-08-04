from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import splu

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio
from loudspeaker_axisym_fem.stage4C_acoustic_structure import build_stage4C_acoustic_structure_model
from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered
from loudspeaker_axisym_fem.exterior_field import hk_pressure_from_samples, spl_db_from_pressure_peak

from p2_axisym_solid import (
    build_p2_solid,
    assemble_region_stiffness,
    assemble_p2_G,
    assemble_nonconforming_p2_G,
    assemble_lorentz_force,
    complex_stiffness,
)
from p2_pml_operator import LocalP2PMLOperator
from global_p2_acoustic_operator import GlobalP2AcousticOperator
from boundary93_parity import Boundary93ParityCorrection
from native_blocked_coil import NativeBlockedCoil


@dataclass
class BestModel:
    root: Path
    mesh: object
    acoustic_model: object
    solid: object
    region_stiffness: dict
    G: csr_matrix
    G_info: dict
    lorentz_per_A: np.ndarray
    lorentz_info: dict
    acoustic_operator: LocalP2PMLOperator | GlobalP2AcousticOperator
    config: dict
    blocked_coil: NativeBlockedCoil | None = None


@dataclass
class FrequencySolution:
    freq_Hz: float
    current_A_peak: complex
    voltage_V_peak: complex
    blocked_impedance_ohm: complex | None
    motional_impedance_ohm: complex
    total_impedance_ohm: complex | None
    solid_displacement: np.ndarray
    pressure_mixed: np.ndarray
    pressure_base: np.ndarray
    p_axis_1m_Pa_peak: complex
    axis_SPL_dB: float
    directivity_angles_deg: np.ndarray
    directivity_pressure_Pa_peak: np.ndarray
    directivity_relative_dB: np.ndarray
    pml_info: dict
    metadata: dict


def load_config(root: str | Path, config_path: str | Path | None = None) -> dict:
    root = Path(root)
    p = Path(config_path) if config_path else root / "configs" / "best_model.json"
    if not p.is_absolute():
        p = root / p

    def merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key == "extends":
                continue
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    data = json.loads(p.read_text(encoding="utf-8"))
    parent = data.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = p.parent / parent_path
        return merge(load_config(root, parent_path), data)
    return data


def build_best_model(
    root: str | Path,
    *,
    config_path: str | Path | None = None,
    magnetostatic_vtu: str | Path | None = None,
    build_blocked_coil: bool = True,
) -> BestModel:
    root = Path(root).resolve()
    cfg = load_config(root, config_path)
    mesh_path = root / cfg["geometry"]["mesh"]
    structure_mesh_path = root / cfg["geometry"].get("structure_mesh", cfg["geometry"]["mesh"])
    mphtxt = root / cfg["geometry"]["mphtxt"]
    mesh = load_tagged_meshio(mesh_path)
    structure_mesh = mesh if structure_mesh_path == mesh_path else load_tagged_meshio(structure_mesh_path)
    ac = build_stage4C_acoustic_structure_model(mesh, mphtxt, solid_uniform_refine=0, c0=cfg["air"]["c0_m_s"])
    solid = build_p2_solid(structure_mesh)
    regions = assemble_region_stiffness(solid)
    if magnetostatic_vtu is None:
        magnetostatic_vtu = root / "runs" / "magnetics" / "magnetostatic_solution.vtu"
    magnetostatic_vtu = Path(magnetostatic_vtu)
    if not magnetostatic_vtu.exists():
        raise FileNotFoundError(
            f"magnetostatic field not found: {magnetostatic_vtu}. "
            "Run `python cli.py magnetics` first or pass --magnetostatic-vtu."
        )
    g, ginfo = assemble_lorentz_force(solid, magnetostatic_vtu)
    acoustics_cfg = cfg.get("acoustics", {})
    acoustic_order = int(acoustics_cfg.get("physical_pressure_order", 1))
    selective_p2_domains = list(map(int, acoustics_cfg.get("selective_p2_domains", [])))
    preserve_p1_pml_trace = bool(acoustics_cfg.get("preserve_p1_pml_interface_trace", True))
    operator_class = (
        GlobalP2AcousticOperator
        if acoustic_order == 2 and not preserve_p1_pml_trace
        else LocalP2PMLOperator
    )
    op = operator_class(
        ac,
        None,
        c0_m_s=cfg["air"]["c0_m_s"],
        rho0_kg_m3=cfg["air"]["rho0_kg_m3"],
        quadrature_order=cfg["pml"]["quadrature_order"],
        **({"physical_p2_domains": selective_p2_domains} if operator_class is LocalP2PMLOperator else {}),
    )
    use_p2_trace = acoustic_order == 2 or bool(selective_p2_domains)
    assemble_G = assemble_p2_G if structure_mesh is mesh else assemble_nonconforming_p2_G
    G, G_info = assemble_G(ac, solid, pressure_operator=op if use_p2_trace else None)
    blocker = None
    bcfg = cfg.get("blocked_coil", {})
    if build_blocked_coil and str(bcfg.get("mode", "")).startswith("native"):
        if bcfg.get("runtime_mode") == "embedded_native_surrogate":
            blocker = NativeBlockedCoil.surrogate_only(mesh, bcfg)
        else:
            blocked_mesh = mesh
            if bcfg.get("field_mesh"):
                blocked_mesh = load_tagged_meshio(root / bcfg["field_mesh"])
            blocked_vtu = root / bcfg.get(
                "magnetostatic_vtu",
                "inputs/comsol_reference/magnetostatic_converged_55iter.vtu",
            )
            blocker = NativeBlockedCoil.from_vtu(blocked_mesh, blocked_vtu, bcfg)
    return BestModel(root, mesh, ac, solid, regions, G, G_info, g, ginfo, op, cfg, blocker)


def _interpolate_blocked_impedance(csv_path: str | Path, freq_Hz: float) -> complex:
    df = pd.read_csv(csv_path)
    fcol = next((c for c in ["freq_Hz", "f_Hz", "frequency_Hz"] if c in df.columns), None)
    if fcol is None:
        raise KeyError("blocked impedance CSV needs freq_Hz or f_Hz")
    pairs = [
        ("Z_blocked_real_ohm", "Z_blocked_imag_ohm"),
        ("Zb_real_ohm", "Zb_imag_ohm"),
        ("Z_real_ohm", "Z_imag_ohm"),
    ]
    pair = next((p for p in pairs if set(p).issubset(df.columns)), None)
    if pair is None:
        raise KeyError("blocked impedance CSV needs complex real/imag columns")
    x = np.log(df[fcol].to_numpy(float))
    xf = math.log(float(freq_Hz))
    return complex(np.interp(xf, x, df[pair[0]]), np.interp(xf, x, df[pair[1]]))


def _exterior_field(model: BestModel, freq_Hz: float, pressure: np.ndarray):
    cfg = model.config
    ext = cfg["exterior"]
    if hasattr(model.acoustic_operator, "boundary_samples"):
        samples, info = model.acoustic_operator.boundary_samples(
            pressure,
            boundary_id=int(ext.get("boundary_id", 93)),
            intorder=4,
            force_radial_normals=bool(ext.get("force_radial_normals", True)),
        )
    else:
        samples, info = boundary93_hk_samples_recovered(
            model.acoustic_model, model.acoustic_operator.base_pressure(pressure),
            recovery_method="ppr" if str(ext.get("recovery_method", "ppr")).startswith("ppr") else "zz",
            force_radial_normals=bool(ext.get("force_radial_normals", True)),
        )
    parity_info = {"applied_to_hk": False}
    parity_cfg = ext.get("req6_ppr_parity", {})
    if bool(parity_cfg.get("apply_to_hk", False)):
        parity = Boundary93ParityCorrection.from_json(model.root / parity_cfg["config"])
        rs, zs, nr, nz, ds, ps, qs = samples
        qs = parity.apply(freq_Hz, rs, zs, ps, qs)
        samples = (rs, zs, nr, nz, ds, ps, qs)
        parity_info = {"applied_to_hk": True, "kind": parity.config.get("kind")}
    amin, amax, n = ext["angles_deg"]
    angles = np.linspace(float(amin), float(amax), int(n))
    theta = np.deg2rad(angles)
    radius = float(ext["observation_radius_m"])
    obs_r = np.abs(np.sin(theta)) * radius
    obs_z = np.cos(theta) * radius
    p = hk_pressure_from_samples(
        freq_Hz,
        cfg["air"]["c0_m_s"],
        *samples,
        obs_r=obs_r,
        obs_z=obs_z,
        nphi=int(ext["azimuth_quadrature_points"]),
        mirror=bool(ext["mirror_sound_hard_plane"]),
        sign=-1,
    )
    i0 = int(np.argmin(np.abs(angles)))
    relative = 20 * np.log10(np.maximum(np.abs(p) / max(abs(p[i0]), 1e-300), 1e-300))
    return angles, p, relative, info, parity_info


def solve_frequency(
    model: BestModel,
    freq_Hz: float,
    *,
    drive: str = "current",
    current_A_peak: complex = 1.0 + 0j,
    voltage_V_peak: complex = 1.0 + 0j,
    blocked_impedance_csv: str | Path | None = None,
    nra_enabled: bool = True,
) -> FrequencySolution:
    f = float(freq_Hz)
    w = 2 * math.pi * f
    sf = model.solid.free_dofs
    K = complex_stiffness(
        model.solid, w, f, model.region_stiffness,
        high_frequency_multipliers=model.config.get("structure", {}).get("high_frequency_stiffness_multipliers_relative_to_reference"),
        high_frequency_loss_multipliers=model.config.get("structure", {}).get("high_frequency_loss_multipliers_relative_to_reference"),
    )
    H = (K[sf][:, sf] - w * w * model.solid.M[sf][:, sf]).tocsr()
    Aac, pml_info = model.acoustic_operator.matrix(f, nra_enabled=nra_enabled)
    G = model.G
    if G.shape[1] == model.acoustic_operator.n2:
        Gext = G
    else:
        Gext = bmat([[G, csr_matrix((model.solid.ndof, model.acoustic_operator.n2 - G.shape[1]))]], format="csr")
    system = bmat([
        [H, -Gext[sf, :]],
        [-model.config["air"]["rho0_kg_m3"] * w * w * Gext.T[:, sf], Aac],
    ], format="csc")
    rhs = np.r_[model.lorentz_per_A[sf], np.zeros(model.acoustic_operator.n2, complex)]
    unit = splu(system).solve(rhs)
    u_unit = np.zeros(model.solid.ndof, complex)
    u_unit[sf] = unit[: len(sf)]
    p_unit = unit[len(sf):]
    z_motional = 1j * w * np.vdot(model.lorentz_per_A, u_unit)

    Zb = None
    Ztotal = None
    if blocked_impedance_csv is not None:
        Zb = _interpolate_blocked_impedance(blocked_impedance_csv, f)
        blocked_source = "external_csv_compatibility"
    elif model.blocked_coil is not None:
        Zb = model.blocked_coil.impedance(f)
        blocked_source = "native_voltage_constrained_reference_identified"
    else:
        blocked_source = "none"
    if drive == "voltage":
        if Zb is None:
            raise ValueError(
                "voltage drive requires native blocked_coil configuration "
                "or --blocked-impedance-csv"
            )
        Ztotal = Zb + z_motional
        current = voltage_V_peak / Ztotal
        voltage = voltage_V_peak
    elif drive == "current":
        current = current_A_peak
        voltage = complex(np.nan, np.nan)
        if Zb is not None:
            Ztotal = Zb + z_motional
            voltage = current * Ztotal
    else:
        raise ValueError("drive must be current or voltage")

    u = u_unit * current
    p = p_unit * current
    p_base = model.acoustic_operator.base_pressure(p)
    angles, pdir, rel, binfo, parity_info = _exterior_field(model, f, p)
    i0 = int(np.argmin(np.abs(angles)))
    p_axis = pdir[i0]
    spl = float(spl_db_from_pressure_peak(p_axis, model.config["air"]["p_ref_Pa"]))
    metadata = {
        "solid": model.solid.summary(),
        "G": model.G_info,
        "lorentz": model.lorentz_info,
        "boundary93": {
            **binfo.__dict__,
            "recovery_method": model.config["exterior"].get("recovery_method", "ppr"),
            "force_radial_normals": model.config["exterior"].get("force_radial_normals", True),
            "phasor_convention": model.config["exterior"].get("phasor_convention"),
            "req6_parity": parity_info,
        },
        "drive": drive,
        "blocked_impedance_source": blocked_source,
        "nra_enabled": bool(nra_enabled),
    }
    return FrequencySolution(f, current, voltage, Zb, z_motional, Ztotal, u, p, p_base, p_axis, spl, angles, pdir, rel, pml_info, metadata)


def solve_sweep(
    model: BestModel,
    frequencies_Hz: Iterable[float],
    **kwargs,
) -> list[FrequencySolution]:
    return [solve_frequency(model, float(f), **kwargs) for f in frequencies_Hz]
