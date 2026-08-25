#!/usr/bin/env python3
"""Audit production wet-trace candidates and make temporary geometry previews.

The production input is read only.  Optional reference meshes are generated
under a caller-selected temporary directory solely for plotting; no mesh,
plot, or solver result is written into the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loudspeaker_axisym_fem.enclosure_geometry import (  # noqa: E402
    REFERENCE_CONFIGS,
    generate_reference_mesh,
)
from loudspeaker_axisym_fem.production_wet_trace import (  # noqa: E402
    DEFAULT_MPHTXT,
    DEFAULT_PRODUCTION_MESH,
    extract_production_wet_traces,
    write_json,
)


def _domain_names(mesh: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in mesh.field_data.items():
        if int(value[1]) == 2:
            result[int(value[0])] = str(name)
    return result


def _line_names(mesh: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in mesh.field_data.items():
        if int(value[1]) == 1:
            result[int(value[0])] = str(name)
    return result


def _reference_preview(mesh_path: Path, case_id: str, output_path: Path) -> Path:
    """Plot a readable r-z overview of one reference L0 mesh."""

    import meshio

    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    triangle_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    domains = _domain_names(mesh)
    boundaries = _line_names(mesh)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.tripcolor(
            points[:, 0],
            points[:, 1],
            triangles,
            facecolors=triangle_tags,
            shading="flat",
            cmap="tab20",
            alpha=0.62,
        )
        ax.triplot(
            points[:, 0],
            points[:, 1],
            triangles,
            color="0.25",
            linewidth=0.16,
            alpha=0.36,
        )
        for block in mesh.cells:
            if block.type != "line":
                continue
            tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            # meshio stores the line blocks contiguously in the common case;
            # this mesh has one line block, and the fallback index is explicit.
            for segment, tag in zip(block.data, tags):
                name = boundaries.get(int(tag), f"boundary_{int(tag)}")
                color = "black" if name == "axis" else "0.08"
                alpha = 0.90 if name == "axis" else 0.18
                ax.plot(
                    points[segment, 0],
                    points[segment, 1],
                    color=color,
                    linewidth=0.50 if name == "axis" else 0.22,
                    alpha=alpha,
                )
        ax.axvline(0.0, color="black", linewidth=0.9, label="axis r=0")
        handles = [
            Patch(facecolor=plt.get_cmap("tab20")(i % 20), alpha=0.62, label=name)
            for i, name in enumerate(sorted(domains.values()))
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{case_id} L0 enclosure reference geometry; reference planar piston")
        ax.grid(True, linewidth=0.25, alpha=0.35)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except ImportError:
        return _svg_fallback(points, triangles, output_path, f"{case_id} L0 reference geometry")


def _production_preview(report: dict[str, Any], output_path: Path) -> Path:
    """Plot the production mesh with discovered front/rear traces highlighted."""

    import meshio

    mesh = meshio.read(report["source"]["mesh_path"])
    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.triplot(
            points[:, 0],
            points[:, 1],
            triangles,
            color="0.30",
            linewidth=0.16,
            alpha=0.24,
        )
        for side, color in (("front", "tab:red"), ("rear", "tab:blue")):
            for entity in report[side]["entities"]:
                for polyline in entity["polylines_rz"]:
                    values = list(zip(*polyline))
                    ax.plot(
                        values[0],
                        values[1],
                        color=color,
                        linewidth=2.0,
                        label=f"{side} boundary {entity['boundary_id']}",
                    )
        ax.axvline(0.0, color="black", linewidth=0.9, label="axis r=0")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_title("Production wet-trace candidates; not production-interface ready")
        ax.grid(True, linewidth=0.25, alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except ImportError:
        return _svg_fallback(points, triangles, output_path.with_suffix(".svg"), "production wet traces")


def _svg_fallback(points: Any, triangles: Any, output_path: Path, title: str) -> Path:
    """Minimal vector fallback if matplotlib is unavailable."""

    import numpy as np

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    width, height = 1200, 800

    def xy(point: Any) -> tuple[float, float]:
        x = 30.0 + 1140.0 * (float(point[0]) - xmin) / max(xmax - xmin, 1e-15)
        y = 30.0 + 740.0 * (ymax - float(point[1])) / max(ymax - ymin, 1e-15)
        return x, y

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<title>{title}</title>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for triangle in triangles:
        p = [xy(points[index]) for index in triangle]
        rows.append(
            '<polygon points="%s" fill="none" stroke="#777" stroke-width="0.25"/>'
            % " ".join(f"{x:.3f},{y:.3f}" for x, y in p)
        )
    rows.append(f'<text x="30" y="20" font-size="16">{title}</text>')
    rows.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=ROOT / DEFAULT_PRODUCTION_MESH)
    parser.add_argument("--mphtxt", type=Path, default=ROOT / DEFAULT_MPHTXT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/luna_production_wet_trace_audit.json"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="temporary directory for the production and A-E L0 previews",
    )
    parser.add_argument(
        "--reference-mesh-dir",
        type=Path,
        default=Path("/tmp/luna_enclosure_phase2_reference_meshes"),
    )
    parser.add_argument(
        "--generate-reference",
        action="store_true",
        help="generate A-E L0 meshes under --reference-mesh-dir for previews",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = extract_production_wet_traces(args.mesh, mphtxt_path=args.mphtxt)
    plot_paths: list[str] = []
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        production_path = _production_preview(report, args.plot_dir / "production_wet_trace.png")
        plot_paths.append(str(production_path))
        for case_id, config_name in sorted(REFERENCE_CONFIGS.items()):
            mesh_path = args.reference_mesh_dir / f"{case_id}_L0.msh"
            if args.generate_reference or not mesh_path.exists():
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                generate_reference_mesh(ROOT / "configs" / "enclosures" / config_name, "L0", mesh_path)
            plot_paths.append(str(_reference_preview(mesh_path, case_id, args.plot_dir / f"{case_id}_L0.png")))
    report["plots"] = {
        "directory": str(args.plot_dir) if args.plot_dir is not None else None,
        "paths": plot_paths,
    }
    write_json(report, args.output)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

