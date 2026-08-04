from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from loudspeaker_axisym_fem.stage4F_hk_refinement import boundary93_hk_samples_recovered
from loudspeaker_axisym_fem.exterior_field import hk_pressure_from_samples


def _complex_columns(prefix: str, x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_real": np.real(x),
        f"{prefix}_imag": np.imag(x),
        f"{prefix}_abs": np.abs(x),
        f"{prefix}_phase_deg": np.angle(x, deg=True),
    }


def write_solution_files(model, solution, outdir: str | Path) -> dict:
    import meshio

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ftag = f"{solution.freq_Hz:g}Hz"

    npz_path = out / f"solution_{ftag}.npz"
    np.savez_compressed(
        npz_path,
        freq_Hz=np.array(solution.freq_Hz),
        current_A_peak=np.array(solution.current_A_peak),
        voltage_V_peak=np.array(solution.voltage_V_peak),
        blocked_impedance_ohm=np.array(solution.blocked_impedance_ohm if solution.blocked_impedance_ohm is not None else np.nan + 1j * np.nan),
        motional_impedance_ohm=np.array(solution.motional_impedance_ohm),
        total_impedance_ohm=np.array(solution.total_impedance_ohm if solution.total_impedance_ohm is not None else np.nan + 1j * np.nan),
        solid_displacement=solution.solid_displacement,
        pressure_mixed=solution.pressure_mixed,
        pressure_base=solution.pressure_base,
        directivity_angles_deg=solution.directivity_angles_deg,
        directivity_pressure_Pa_peak=solution.directivity_pressure_Pa_peak,
        directivity_relative_dB=solution.directivity_relative_dB,
    )

    spts = np.column_stack([model.solid.points_rz_m, np.zeros(len(model.solid.points_rz_m))])
    ur = solution.solid_displacement[0::2]
    uz = solution.solid_displacement[1::2]
    smesh = meshio.Mesh(
        points=spts,
        cells=[("triangle6", model.solid.triangles6)],
        point_data={
            **_complex_columns("u_r_m", ur),
            **_complex_columns("u_z_m", uz),
            "u_abs_m": np.sqrt(np.abs(ur) ** 2 + np.abs(uz) ** 2),
        },
        cell_data={"domain_id": [model.solid.domains.astype(int)]},
    )
    solid_vtu = out / f"solid_{ftag}.vtu"
    meshio.write(solid_vtu, smesh)

    apts2, physical, pml6 = model.acoustic_operator.mixed_points_and_cells()
    apts = np.column_stack([apts2, np.zeros(len(apts2))])
    pvals = model.acoustic_operator.pressure_for_mixed_points(solution.pressure_mixed)
    cells = []
    cell_data = []
    if len(physical):
        cells.append(("triangle", physical))
        cell_data.append(np.zeros(len(physical), int))
    if len(pml6):
        cells.append(("triangle6", pml6))
        if hasattr(model.acoustic_operator, "mixed_triangle6_domains"):
            domains6 = model.acoustic_operator.mixed_triangle6_domains()
            cell_data.append(np.isin(domains6, (3, 4)).astype(int))
        else:
            cell_data.append(np.ones(len(pml6), int))
    amesh = meshio.Mesh(
        points=apts,
        cells=cells,
        point_data={
            **_complex_columns("p_Pa_peak", pvals),
            "SPL_dB": 20 * np.log10(np.maximum(np.abs(pvals) / math.sqrt(2), 1e-300) / model.config["air"]["p_ref_Pa"]),
        },
        cell_data={"is_PML": cell_data},
    )
    acoustic_vtu = out / f"acoustic_{ftag}.vtu"
    meshio.write(acoustic_vtu, amesh)

    ddf = pd.DataFrame({
        "theta_deg": solution.directivity_angles_deg,
        "p_real_Pa": solution.directivity_pressure_Pa_peak.real,
        "p_imag_Pa": solution.directivity_pressure_Pa_peak.imag,
        "SPL_dB": 20 * np.log10(np.maximum(np.abs(solution.directivity_pressure_Pa_peak) / math.sqrt(2), 1e-300) / model.config["air"]["p_ref_Pa"]),
        "relative_dB": solution.directivity_relative_dB,
    })
    directivity_csv = out / f"directivity_{ftag}.csv"
    ddf.to_csv(directivity_csv, index=False)

    summary = {
        "freq_Hz": solution.freq_Hz,
        "current_A_peak": [solution.current_A_peak.real, solution.current_A_peak.imag],
        "voltage_V_peak": [solution.voltage_V_peak.real, solution.voltage_V_peak.imag],
        "motional_impedance_ohm": [solution.motional_impedance_ohm.real, solution.motional_impedance_ohm.imag],
        "total_impedance_ohm": None if solution.total_impedance_ohm is None else [solution.total_impedance_ohm.real, solution.total_impedance_ohm.imag],
        "p_axis_1m_Pa_peak": [solution.p_axis_1m_Pa_peak.real, solution.p_axis_1m_Pa_peak.imag],
        "axis_SPL_dB": solution.axis_SPL_dB,
        "pml": solution.pml_info,
        "metadata": solution.metadata,
        "files": {
            "npz": npz_path.name,
            "solid_vtu": solid_vtu.name,
            "acoustic_vtu": acoustic_vtu.name,
            "directivity_csv": directivity_csv.name,
        },
    }
    summary_path = out / f"summary_{ftag}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"npz": npz_path, "solid_vtu": solid_vtu, "acoustic_vtu": acoustic_vtu, "directivity_csv": directivity_csv, "summary": summary_path}


def _subdivided_acoustic_triangles(model):
    points, physical, pml6 = model.acoustic_operator.mixed_points_and_cells()
    tri = [list(map(int, x)) for x in physical]
    for c in pml6:
        a, b, c0, ab, bc, ca = map(int, c)
        tri.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c0], [ab, bc, ca]])
    return points, np.asarray(tri, int)


def render_solution(model, solution, outdir: str | Path, *, exterior_grid: bool = True, exterior_r_points: int = 61, exterior_z_points: int = 81, exterior_nphi: int = 48) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ftag = f"{solution.freq_Hz:g}Hz"
    written = []

    # Structural displacement magnitude and phase.
    pts = model.solid.points_rz_m
    tri3 = model.solid.triangles6[:, :3]
    ur = solution.solid_displacement[0::2]
    uz = solution.solid_displacement[1::2]
    umag = np.sqrt(np.abs(ur) ** 2 + np.abs(uz) ** 2)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri3)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.tricontourf(triang, np.log10(np.maximum(umag, 1e-30)), 80)
    fig.colorbar(im, ax=ax, label="log10 |u| [m]")
    ax.set_aspect("equal"); ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Solid displacement magnitude, {ftag}")
    fig.tight_layout(); p = out / f"solid_displacement_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    # Acoustic SPL and phase, including locally enriched PML.
    apoints, atri = _subdivided_acoustic_triangles(model)
    pvals = model.acoustic_operator.pressure_for_mixed_points(solution.pressure_mixed)
    atriang = mtri.Triangulation(apoints[:, 0], apoints[:, 1], atri)
    spl = 20 * np.log10(np.maximum(np.abs(pvals) / math.sqrt(2), 1e-300) / model.config["air"]["p_ref_Pa"])
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.tricontourf(atriang, spl, 80)
    fig.colorbar(im, ax=ax, label="SPL [dB re 20 µPa]")
    ax.set_aspect("equal"); ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Acoustic/PML full field, {ftag}")
    fig.tight_layout(); p = out / f"acoustic_SPL_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.tricontourf(atriang, np.angle(pvals, deg=True), np.linspace(-180, 180, 73))
    fig.colorbar(im, ax=ax, label="Pressure phase [deg]")
    ax.set_aspect("equal"); ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Acoustic pressure phase, {ftag}")
    fig.tight_layout(); p = out / f"acoustic_phase_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    # Directivity Cartesian and polar.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(solution.directivity_angles_deg, solution.directivity_relative_dB)
    ax.set_xlim(-90, 90); ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Theta [deg]"); ax.set_ylabel("Relative SPL [dB]"); ax.set_title(f"Directivity, {ftag}")
    fig.tight_layout(); p = out / f"directivity_cartesian_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    theta = np.deg2rad(solution.directivity_angles_deg)
    radius = np.maximum(solution.directivity_relative_dB, -60)
    fig = plt.figure(figsize=(6.5, 6.5)); ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, radius); ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_rlim(-60, 0); ax.set_title(f"Directivity polar, {ftag}")
    fig.tight_layout(); p = out / f"directivity_polar_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    if exterior_grid:
        samples, _ = boundary93_hk_samples_recovered(model.acoustic_model, solution.pressure_base)
        r = np.linspace(0.0, 0.60, int(exterior_r_points))
        z = np.linspace(0.17, 1.0, int(exterior_z_points))
        R, Z = np.meshgrid(r, z)
        pe = hk_pressure_from_samples(
            solution.freq_Hz,
            model.config["air"]["c0_m_s"],
            *samples,
            obs_r=R.ravel(),
            obs_z=Z.ravel(),
            nphi=int(exterior_nphi),
            mirror=bool(model.config["exterior"]["mirror_sound_hard_plane"]),
            sign=-1,
        ).reshape(R.shape)
        es = 20 * np.log10(np.maximum(np.abs(pe) / math.sqrt(2), 1e-300) / model.config["air"]["p_ref_Pa"])
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        im = ax.contourf(R, Z, es, 80)
        fig.colorbar(im, ax=ax, label="Exterior SPL [dB]")
        ax.set_aspect("equal"); ax.set_xlabel("r [m]"); ax.set_ylabel("z [m]"); ax.set_title(f"Exterior full field, {ftag}")
        fig.tight_layout(); p = out / f"exterior_SPL_{ftag}.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)

    return written


def write_sweep_metrics(solutions, outdir: str | Path) -> Path:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in solutions:
        rows.append({
            "freq_Hz": s.freq_Hz,
            "current_abs_A_peak": abs(s.current_A_peak),
            "current_phase_deg": np.angle(s.current_A_peak, deg=True),
            "Zmot_real_ohm": s.motional_impedance_ohm.real,
            "Zmot_imag_ohm": s.motional_impedance_ohm.imag,
            "Zmot_abs_ohm": abs(s.motional_impedance_ohm),
            "Ztotal_real_ohm": np.nan if s.total_impedance_ohm is None else s.total_impedance_ohm.real,
            "Ztotal_imag_ohm": np.nan if s.total_impedance_ohm is None else s.total_impedance_ohm.imag,
            "Ztotal_abs_ohm": np.nan if s.total_impedance_ohm is None else abs(s.total_impedance_ohm),
            "axis_SPL_dB": s.axis_SPL_dB,
            "axis_phase_deg": np.angle(s.p_axis_1m_Pa_peak, deg=True),
        })
    p = out / "sweep_metrics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def render_sweep(solutions, outdir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    f = np.asarray([s.freq_Hz for s in solutions])
    written = []
    if all(s.total_impedance_ohm is not None for s in solutions):
        Z = np.asarray([s.total_impedance_ohm for s in solutions])
        fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.semilogx(f, np.abs(Z)); ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|Z| [ohm]"); ax.set_title("Total impedance")
        fig.tight_layout(); p = out / "sweep_impedance.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)
    spl = np.asarray([s.axis_SPL_dB for s in solutions])
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.semilogx(f, spl); ax.grid(True, which="both", alpha=0.3); ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("SPL at 1 m [dB]"); ax.set_title("Axis response")
    fig.tight_layout(); p = out / "sweep_axis_SPL.png"; fig.savefig(p, dpi=180); plt.close(fig); written.append(p)
    return written
