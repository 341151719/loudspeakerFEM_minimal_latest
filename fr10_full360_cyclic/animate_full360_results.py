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
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def _membrane_boundary_triangles(mesh, part_ids=(0, 1, 2)):
    """Return render triangles on selected tetra/tetra10 membrane exteriors.

    The full-360 result stores surround, cone and dustcap as part IDs 0--2.
    Rendering only tetra nodes produced a point cloud; retaining faces that occur
    once reconstructs the actual outer skin of each solid membrane component.
    A quadratic tetra10 face is split into four triangles so its midside P2
    displacement degrees of freedom also move the rendered surface.
    """
    face_patterns = np.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=int
    )
    boundary_keys = []
    render_blocks = []
    cell_parts = mesh.cell_data.get("part_id")
    if cell_parts is None:
        raise KeyError("structure VTU does not contain cell part_id")
    if len(cell_parts) != len(mesh.cells):
        raise ValueError("cell part_id block count does not match mesh cell blocks")
    # Full-360 reconstruction intentionally keeps separate nodes at cyclic
    # sector seams. Canonical coordinate IDs make coincident seam faces cancel
    # while the part ID prevents cancellation across tied material components.
    _, canonical_point = np.unique(
        np.round(np.asarray(mesh.points, float), 12), axis=0, return_inverse=True
    )
    for block, block_parts in zip(mesh.cells, cell_parts):
        if block.type not in ("tetra", "tetra10"):
            continue
        cells = np.asarray(block.data, dtype=int)
        block_parts = np.asarray(block_parts)
        if len(block_parts) != len(cells):
            raise ValueError(f"cell part_id length mismatch for {block.type} block")
        selected_mask = np.isin(block_parts, part_ids)
        selected_parts = block_parts[selected_mask]
        selected = cells[selected_mask, :4]
        if len(selected):
            corners = np.vstack([selected[:, pattern] for pattern in face_patterns])
            face_parts = np.tile(selected_parts, len(face_patterns))[:, None]
            coordinate_keys = np.sort(canonical_point[corners], axis=1)
            boundary_keys.append(np.hstack((face_parts, coordinate_keys)))
            if block.type == "tetra10":
                selected10 = cells[selected_mask]
                quadratic_patterns = np.asarray(
                    (
                        (0, 2, 1, 6, 5, 4),
                        (0, 1, 3, 4, 8, 7),
                        (1, 2, 3, 5, 9, 8),
                        (2, 0, 3, 6, 7, 9),
                    ),
                    dtype=int,
                )
                quadratic = np.vstack(
                    [selected10[:, pattern] for pattern in quadratic_patterns]
                )
                a, b, c, ab, bc, ca = quadratic.T
                render_blocks.append(
                    np.stack(
                        (
                            np.stack((a, ab, ca), axis=1),
                            np.stack((ab, b, bc), axis=1),
                            np.stack((ca, bc, c), axis=1),
                            np.stack((ab, bc, ca), axis=1),
                        ),
                        axis=1,
                    )
                )
            else:
                render_blocks.append(corners[:, None, :])
    if not boundary_keys:
        raise ValueError("no tetrahedral membrane cells found for part IDs 0, 1, 2")
    keys = np.vstack(boundary_keys)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    exterior = counts[inverse] == 1
    triangles = []
    start = 0
    for candidates in render_blocks:
        stop = start + len(candidates)
        triangles.append(candidates[exterior[start:stop]].reshape(-1, 3))
        start = stop
    return np.vstack(triangles)


def write_membrane_surface_gif(
    structure_vtu,
    output,
    frequency,
    identity,
    frames=24,
    fps=12,
    deformation_scale=None,
    kclass="k0",
    physical_electrical_drive=True,
):
    """Animate the continuous surround/cone/dustcap surface from complex FEM U."""
    mesh = meshio.read(structure_vtu)
    points = np.asarray(mesh.points, float)
    displacement = _complex_point_data(mesh, "u_real_m", "u_imag_m")
    faces = _membrane_boundary_triangles(mesh)
    used = np.unique(faces)
    membrane_amplitude = np.linalg.norm(displacement[used], axis=1)
    max_displacement = max(float(np.max(membrane_amplitude)), 1e-300)
    span = float(np.max(np.ptp(points[used], axis=0)))
    if deformation_scale is None:
        deformation_scale = max(1.0, 0.05 * span / max_displacement)
    deformation_scale = float(deformation_scale)
    uz_limit = max(float(np.max(np.abs(displacement[used, 2]))), 1e-300)
    phases = _phase_angles(frames)

    fig = plt.figure(figsize=(8.7, 7.3))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=27, azim=-58)
    instant = np.real(displacement * np.exp(1j * phases[0]))
    moved = points + deformation_scale * instant
    surface = Poly3DCollection(
        moved[faces] * 1e3,
        cmap="coolwarm",
        linewidths=0.0,
        antialiased=False,
        shade=False,
    )
    surface.set_clim(-uz_limit * 1e6, uz_limit * 1e6)
    surface.set_array(np.mean(instant[faces, 2], axis=1) * 1e6)
    ax.add_collection3d(surface)

    # A fixed dark mounting ring makes the moving 100 mm diaphragm read as a
    # loudspeaker rather than a floating finite-element surface.
    theta = np.linspace(0.0, 2.0 * math.pi, 181)
    used_radius = np.hypot(points[used, 0], points[used, 1])
    outer_radius_m = float(np.max(used_radius))
    outer_nodes = used[used_radius >= 0.985 * outer_radius_m]
    outer_radius = outer_radius_m * 1e3
    ring_z = float(np.median(points[outer_nodes, 2])) * 1e3 - 1.8
    ax.plot(
        outer_radius * np.cos(theta),
        outer_radius * np.sin(theta),
        np.full_like(theta, ring_z),
        color="#222222",
        linewidth=5.0,
        alpha=0.9,
    )

    envelope = points[used].copy()
    envelope[:, 2] += np.sign(envelope[:, 2] + 1e-30) * 0.05 * span
    _equal_3d_axes(ax, envelope * 1e3)
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_zlabel("z / mm")
    ax.set_box_aspect((1.0, 1.0, 0.55))
    title = ax.set_title("")
    colorbar = fig.colorbar(
        ScalarMappable(norm=surface.norm, cmap=surface.cmap),
        ax=ax,
        pad=0.08,
        shrink=0.78,
        label="instantaneous physical u_z / micrometre peak",
    )
    colorbar.ax.tick_params(labelsize=8)

    def update(frame):
        instant_u = np.real(displacement * np.exp(1j * phases[frame]))
        moved_u = points + deformation_scale * instant_u
        surface.set_verts(moved_u[faces] * 1e3)
        surface.set_array(np.mean(instant_u[faces, 2], axis=1) * 1e6)
        title.set_text(
            f"FR10 continuous 3-D diaphragm, {frequency:g} Hz, {identity}\n"
            f"phase {360 * frame / frames:.0f} deg; geometry deformation x{deformation_scale:.3g}"
        )
        return surface, title

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_gif(fig, update, frames, fps, output)
    return {
        "path": str(output),
        "frames": int(frames),
        "fps": int(fps),
        "frequency_Hz": float(frequency),
        "kclass": kclass,
        "physical_electrical_drive": bool(physical_electrical_drive),
        "identity": identity,
        "surface_triangles": int(len(faces)),
        "deformation_scale": deformation_scale,
        "maximum_physical_displacement_um_peak": max_displacement * 1e6,
        "colour_quantity": "instantaneous physical u_z in micrometres peak",
        "surface_parts": ["surround", "cone", "dustcap"],
    }


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


def generate_membrane_surface_animations(results_root, frames=24, fps=12):
    """Render the physical piston/breakup cases plus the k=1 rocking diagnostic."""
    import json

    results_root = Path(results_root).resolve()
    output = results_root / "animations" / "membrane_surface_3d"
    cases = (
        (
            90.0,
            "k=0 physical electrical piston motion",
            results_root
            / "final_baseline/90Hz/structure_full360_90Hz_k0_P2.vtu",
            output / "membrane_surface_90Hz_k0_physical.gif",
            "k0",
            True,
        ),
        (
            2000.0,
            "k=0 physical electrical breakup motion",
            results_root
            / "final_baseline/2000Hz/structure_full360_2000Hz_k0_P2.vtu",
            output / "membrane_surface_2000Hz_k0_physical.gif",
            "k0",
            True,
        ),
        (
            2000.0,
            "k=1 / m=1 unit-force rocking diagnostic",
            results_root
            / "phase_diagnostics/2000Hz_k1/structure_full360_2000Hz_diagnostic_k1_P2.vtu",
            output / "membrane_surface_2000Hz_k1_diagnostic.gif",
            "k1_m1",
            False,
        ),
    )
    missing = [str(source) for _, _, source, _, _, _ in cases if not source.exists()]
    if missing:
        raise FileNotFoundError("missing full-360 structure files: " + ", ".join(missing))
    rows = {}
    for frequency, identity, source, destination, kclass, physical_drive in cases:
        key = destination.stem
        rows[key] = write_membrane_surface_gif(
            source,
            destination,
            frequency,
            identity,
            frames,
            fps,
            kclass=kclass,
            physical_electrical_drive=physical_drive,
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "surface_animation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate FR10 full-360 complex FEM fields")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=2000.0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--surface-suite",
        action="store_true",
        help="render continuous 3-D diaphragm surfaces at 90/2000 Hz",
    )
    args = parser.parse_args()
    if args.surface_suite:
        generate_membrane_surface_animations(args.results_root, args.frames, args.fps)
    else:
        generate_animations(args.results_root, args.frequency, args.frames, args.fps)
