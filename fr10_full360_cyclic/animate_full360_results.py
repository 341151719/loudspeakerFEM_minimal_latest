from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.interpolate import griddata


def _complex_point_data(mesh, real_name, imag_name):
    return np.asarray(mesh.point_data[real_name]) + 1j * np.asarray(
        mesh.point_data[imag_name]
    )


def _sample_indices(count, maximum):
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum, dtype=int)


def _equal_3d_axes(ax, points):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _phase_angles(frames):
    return np.linspace(0.0, 2.0 * math.pi, frames, endpoint=False)


def _save_gif(fig, update, frames, fps, path):
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=115)
    plt.close(fig)


def write_rocking_gif(structure_vtu, output, frequency, frames=24, fps=12):
    mesh = meshio.read(structure_vtu)
    points = np.asarray(mesh.points, float)
    displacement = _complex_point_data(mesh, "u_real_m", "u_imag_m")
    selected = _sample_indices(len(points), 14000)
    points = points[selected]
    displacement = displacement[selected]
    amplitude = np.linalg.norm(displacement, axis=1)
    span = float(np.max(np.ptp(points, axis=0)))
    deformation_scale = 0.08 * span / max(float(np.max(amplitude)), 1e-300)
    uz_limit = max(float(np.max(np.abs(displacement[:, 2]))), 1e-300)
    phases = _phase_angles(frames)

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    instant = np.real(displacement * np.exp(1j * phases[0]))
    moved = points + deformation_scale * instant
    scatter = ax.scatter(
        moved[:, 0] * 1e3,
        moved[:, 1] * 1e3,
        moved[:, 2] * 1e3,
        c=instant[:, 2] * 1e6,
        cmap="coolwarm",
        vmin=-uz_limit * 1e6,
        vmax=uz_limit * 1e6,
        s=1.6,
    )
    _equal_3d_axes(ax, moved * 1e3)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_zlabel("z / mm")
    title = ax.set_title("")
    fig.colorbar(
        scatter,
        ax=ax,
        pad=0.08,
        label="instantaneous u_z / micrometre (unit-force diagnostic)",
    )

    def update(frame):
        instant_u = np.real(displacement * np.exp(1j * phases[frame]))
        moved_u = points + deformation_scale * instant_u
        scatter._offsets3d = (
            moved_u[:, 0] * 1e3,
            moved_u[:, 1] * 1e3,
            moved_u[:, 2] * 1e3,
        )
        scatter.set_array(instant_u[:, 2] * 1e6)
        title.set_text(
            f"FR10 k=1 / m=1 rocking-type motion, {frequency:g} Hz\n"
            f"phase {360 * frame / frames:.0f} deg; deformation x{deformation_scale:.3g}"
        )
        return scatter, title

    _save_gif(fig, update, frames, fps, output)
    return {
        "path": str(output),
        "frames": int(frames),
        "fps": int(fps),
        "deformation_scale": float(deformation_scale),
        "interpretation": "normalized k=1 Bloch diagnostic; deformation is exaggerated",
    }


def _deduplicate_plane(points, values):
    keys = np.round(points, 10)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros(len(unique), complex)
    counts = np.zeros(len(unique), int)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1)
    return unique, sums / counts


def _read_acoustic_pair(front_vtu, rear_vtu):
    result = []
    for path in (front_vtu, rear_vtu):
        mesh = meshio.read(path)
        points = np.asarray(mesh.points, float)
        pressure = _complex_point_data(mesh, "p_real_Pa", "p_imag_Pa")
        result.append((points, pressure))
    return result


def write_meridional_propagation_gif(
    front_vtu, rear_vtu, output, frequency, frames=24, fps=12
):
    domains = _read_acoustic_pair(front_vtu, rear_vtu)
    plane_domains = []
    for points, pressure in domains:
        on_plane = np.abs(points[:, 1]) < 1e-10
        plane, field = _deduplicate_plane(points[on_plane][:, [0, 2]], pressure[on_plane])
        plane_domains.append((plane, field))

    axis = np.linspace(-0.3, 0.3, 360)
    xgrid, zgrid = np.meshgrid(axis, axis)
    interpolated = []
    for points, pressure in plane_domains:
        real = griddata(points, pressure.real, (xgrid, zgrid), method="linear")
        imag = griddata(points, pressure.imag, (xgrid, zgrid), method="linear")
        interpolated.append(real + 1j * imag)
    front, rear = interpolated
    field = np.where(zgrid >= 0.0, front, rear)
    outside_sphere = xgrid * xgrid + zgrid * zgrid > 0.302**2
    field[outside_sphere] = np.nan
    amplitude_limit = max(float(np.nanmax(np.abs(field))), 1e-300)
    phases = _phase_angles(frames)

    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    instant = np.real(field * np.exp(1j * phases[0]))
    image = ax.imshow(
        instant,
        extent=(-300, 300, -300, 300),
        origin="lower",
        cmap="coolwarm",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
        interpolation="bilinear",
    )
    ax.fill_between([-50, 50], [-1.5, -1.5], [1.5, 1.5], color="black", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x / mm")
    ax.set_ylabel("z / mm")
    title = ax.set_title("")
    fig.colorbar(image, ax=ax, label="instantaneous pressure / Pa peak")
    fig.tight_layout()

    def update(frame):
        image.set_data(np.real(field * np.exp(1j * phases[frame])))
        title.set_text(
            f"FR10 front/rear exterior sound propagation, {frequency:g} Hz\n"
            f"phasor reconstruction, phase {360 * frame / frames:.0f} deg"
        )
        return image, title

    _save_gif(fig, update, frames, fps, output)
    return {
        "path": str(output),
        "frames": int(frames),
        "fps": int(fps),
        "plane": "y=0 meridional plane",
        "pressure_normalization_Pa_peak": float(amplitude_limit),
    }


def write_outer_field_3d_gif(
    front_vtu, rear_vtu, output, frequency, frames=24, fps=12
):
    domains = _read_acoustic_pair(front_vtu, rear_vtu)
    points = np.vstack([item[0] for item in domains])
    pressure = np.concatenate([item[1] for item in domains])
    radius = np.linalg.norm(points, axis=1)
    cutaway = (points[:, 1] <= 0.006) & (radius >= 0.055)
    points = points[cutaway]
    pressure = pressure[cutaway]
    selected = _sample_indices(len(points), 17000)
    points = points[selected]
    pressure = pressure[selected]
    pressure_limit = max(float(np.max(np.abs(pressure))), 1e-300)
    phases = _phase_angles(frames)

    fig = plt.figure(figsize=(8.7, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    instant = np.real(pressure * np.exp(1j * phases[0]))
    scatter = ax.scatter(
        points[:, 0] * 1e3,
        points[:, 1] * 1e3,
        points[:, 2] * 1e3,
        c=instant,
        cmap="coolwarm",
        vmin=-pressure_limit,
        vmax=pressure_limit,
        s=2.0,
        alpha=0.82,
    )
    _equal_3d_axes(ax, points * 1e3)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_zlabel("z / mm")
    title = ax.set_title("")
    fig.colorbar(scatter, ax=ax, pad=0.08, label="instantaneous pressure / Pa peak")

    def update(frame):
        scatter.set_array(np.real(pressure * np.exp(1j * phases[frame])))
        title.set_text(
            f"FR10 3-D exterior acoustic cutaway, {frequency:g} Hz\n"
            f"phase {360 * frame / frames:.0f} deg"
        )
        return scatter, title

    _save_gif(fig, update, frames, fps, output)
    return {
        "path": str(output),
        "frames": int(frames),
        "fps": int(fps),
        "pressure_normalization_Pa_peak": float(pressure_limit),
        "view": "3-D half-space cutaway of front and rear exterior domains",
    }


def write_outer_boundary_gif(
    front_vtu, rear_vtu, output, frequency, identity, frames=24, fps=12
):
    domains = _read_acoustic_pair(front_vtu, rear_vtu)
    selected_points = []
    selected_pressure = []
    for points, pressure in domains:
        radius = np.linalg.norm(points, axis=1)
        outer = radius >= 0.995 * float(np.max(radius))
        selected_points.append(points[outer])
        selected_pressure.append(pressure[outer])
    points = np.vstack(selected_points)
    pressure = np.concatenate(selected_pressure)
    limit = max(float(np.max(np.abs(pressure))), 1e-300)
    phases = _phase_angles(frames)

    fig = plt.figure(figsize=(8.5, 7.1))
    ax = fig.add_subplot(111, projection="3d")
    instant = np.real(pressure * np.exp(1j * phases[0]))
    scatter = ax.scatter(
        points[:, 0] * 1e3,
        points[:, 1] * 1e3,
        points[:, 2] * 1e3,
        c=instant,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=8,
    )
    ax.set_xlim(-305, 305)
    ax.set_ylim(-305, 305)
    ax.set_zlim(-305, 305)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_zlabel("z / mm")
    title = ax.set_title("")
    fig.colorbar(scatter, ax=ax, pad=0.08, label="instantaneous pressure / Pa peak")

    def update(frame):
        scatter.set_array(np.real(pressure * np.exp(1j * phases[frame])))
        title.set_text(
            f"FR10 outer boundary pressure, {frequency:g} Hz, {identity}\n"
            f"phase {360 * frame / frames:.0f} deg; R=0.3 m"
        )
        return scatter, title

    _save_gif(fig, update, frames, fps, output)
    return {
        "path": str(output),
        "frames": int(frames),
        "fps": int(fps),
        "pressure_limit_Pa_peak": float(limit),
        "boundary": "R=0.3 m first-order Sommerfeld Robin boundary",
    }


def generate_animations(results_root, frequency=2000.0, frames=24, fps=12):
    results_root = Path(results_root).resolve()
    tag = f"{frequency:g}Hz"
    baseline = results_root / "final_baseline" / tag
    diagnostic = results_root / "phase_diagnostics" / f"{tag}_k1"
    output = results_root / "animations" / tag
    output.mkdir(parents=True, exist_ok=True)
    structure = diagnostic / f"structure_full360_{tag}_diagnostic_k1_P2.vtu"
    front = baseline / f"acoustic_front_full360_{tag}_k0.vtu"
    rear = baseline / f"acoustic_rear_full360_{tag}_k0.vtu"
    diagnostic_front = (
        diagnostic / f"acoustic_front_full360_{tag}_diagnostic_k1.vtu"
    )
    diagnostic_rear = diagnostic / f"acoustic_rear_full360_{tag}_diagnostic_k1.vtu"
    required = (structure, front, rear, diagnostic_front, diagnostic_rear)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing full-360 result files: " + ", ".join(missing))

    rows = {
        "rocking_vibration": write_rocking_gif(
            structure, output / f"rocking_vibration_{tag}_k1.gif", frequency, frames, fps
        ),
        "source_propagation": write_meridional_propagation_gif(
            front,
            rear,
            output / f"source_propagation_meridional_{tag}_k0.gif",
            frequency,
            frames,
            fps,
        ),
        "outer_field_3d": write_outer_field_3d_gif(
            front,
            rear,
            output / f"outer_field_3d_propagation_{tag}_k0.gif",
            frequency,
            frames,
            fps,
        ),
        "outer_boundary_k0": write_outer_boundary_gif(
            front,
            rear,
            output / f"outer_boundary_pressure_{tag}_k0.gif",
            frequency,
            "k=0 physical electrical baseline",
            frames,
            fps,
        ),
        "outer_boundary_k1": write_outer_boundary_gif(
            diagnostic_front,
            diagnostic_rear,
            output / f"outer_boundary_pressure_{tag}_k1_diagnostic.gif",
            frequency,
            "k=1 / m=1 unit-force diagnostic",
            frames,
            fps,
        ),
    }
    import json

    (output / "animation_summary.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate FR10 full-360 complex FEM fields")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=2000.0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    generate_animations(args.results_root, args.frequency, args.frames, args.fps)
