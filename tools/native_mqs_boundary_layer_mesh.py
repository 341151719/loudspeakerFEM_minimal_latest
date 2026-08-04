#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio, skin_depth_m
from loudspeaker_axisym_fem.comsol_geom_mphtxt import parse_mphtxt


PROFILES = {
    "L0": dict(normal_um=32.0, tangent_um=400.0, thickness_um=350.0, ratio=1.30),
    "L1": dict(normal_um=24.0, tangent_um=320.0, thickness_um=450.0, ratio=1.25),
    "L2": dict(normal_um=16.0, tangent_um=250.0, thickness_um=600.0, ratio=1.22),
    "L3": dict(normal_um=10.0, tangent_um=180.0, thickness_um=750.0, ratio=1.18),
}


def _triangle_quality(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    xy = points[triangles]
    edges = np.linalg.norm(xy - np.roll(xy, 1, axis=1), axis=2)
    v1 = xy[:, 1] - xy[:, 0]
    v2 = xy[:, 2] - xy[:, 0]
    area = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return 4.0 * math.sqrt(3.0) * area / np.maximum(np.sum(edges**2, axis=1), 1e-300)


def generate(args) -> dict:
    profile = dict(PROFILES[args.profile])
    if args.normal_um is not None:
        profile["normal_um"] = args.normal_um
    if args.tangent_um is not None:
        profile["tangent_um"] = args.tangent_um
    if args.thickness_um is not None:
        profile["thickness_um"] = args.thickness_um
    if args.ratio is not None:
        profile["ratio"] = args.ratio

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    geo = out / f"native_mqs_bl_{args.profile}.geo"
    msh = out / f"native_mqs_bl_{args.profile}.msh"

    geom = parse_mphtxt(args.mphtxt)
    boundary_layer_curves = tuple(
        int(x) for x in args.boundary_layer_curves.split(",") if x.strip()
    )
    geom.export_geo_comsol_mesh(
        geo,
        global_hmax_m=args.global_h_mm * 1e-3,
        global_hmin_m=profile["normal_um"] * 1e-6,
        boundary_layer_size_m=profile["normal_um"] * 1e-6,
        boundary_layer_tangent_size_m=profile["tangent_um"] * 1e-6,
        corner_refinement_curves=tuple(
            int(x) for x in args.corner_refinement_curves.split(",") if x.strip()
        ),
        corner_refinement_size_m=args.corner_refinement_um * 1e-6,
        mesh_size_extend_from_boundary=not args.no_extend_from_boundary,
        boundary_layer_thickness_m=profile["thickness_um"] * 1e-6,
        boundary_layer_ratio=profile["ratio"],
        boundary_layer_curves=boundary_layer_curves,
        boundary_layer_target_domains=(6, 23),
        boundary_layer_quads=args.quads,
    )

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", int(args.verbose))
        gmsh.open(str(geo))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()

    tagged = load_tagged_meshio(msh)
    raw_mesh = meshio.read(msh)
    raw_cell_counts: dict[str, int] = {}
    for block in raw_mesh.cells:
        raw_cell_counts[block.type] = raw_cell_counts.get(block.type, 0) + len(block.data)
    quality = _triangle_quality(tagged.points_rz_m, tagged.triangles)
    domain_counts = {
        int(d): int(np.sum(tagged.tri_domains == d))
        for d in np.unique(tagged.tri_domains)
    }
    soft = np.isin(tagged.tri_domains, (6, 23))
    soft_tri = tagged.triangles[soft]
    soft_quality = quality[soft]
    soft_edges = np.linalg.norm(
        tagged.points_rz_m[soft_tri]
        - np.roll(tagged.points_rz_m[soft_tri], 1, axis=1),
        axis=2,
    ).ravel()
    delta = skin_depth_m(args.fmax_Hz, args.sigma_S_m, args.mu_r_design)
    min_edge_to_first_layer = float(
        np.min(soft_edges) / (profile["normal_um"] * 1e-6)
    )
    rejection_reasons = []
    if min_edge_to_first_layer < 0.25:
        rejection_reasons.append(
            "soft-iron minimum edge is below 25% of requested first-layer height; "
            "likely boundary-layer collision/corner collapse"
        )
    if float(np.min(soft_quality)) < 0.015:
        rejection_reasons.append(
            "soft-iron minimum triangle quality is below 0.015"
        )
    summary = {
        "profile": args.profile,
        "parameters": profile,
        "boundary_layer_curves": boundary_layer_curves,
        "corner_refinement_curves": tuple(
            int(x) for x in args.corner_refinement_curves.split(",") if x.strip()
        ),
        "corner_refinement_um": args.corner_refinement_um,
        "mesh_size_extend_from_boundary": not args.no_extend_from_boundary,
        "mesh": str(msh),
        "n_nodes": tagged.n_nodes,
        "n_triangles_after_quad_split": tagged.n_triangles,
        "raw_cell_counts": raw_cell_counts,
        "domain_triangle_counts": domain_counts,
        "soft_iron_edge_quantiles_um": {
            str(q): float(np.quantile(soft_edges, q) * 1e6)
            for q in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)
        },
        "triangle_quality_quantiles": {
            str(q): float(np.quantile(quality, q))
            for q in (0.0, 0.01, 0.1, 0.5)
        },
        "soft_iron_triangle_quality_quantiles": {
            str(q): float(np.quantile(soft_quality, q))
            for q in (0.0, 0.01, 0.1, 0.5)
        },
        "minimum_soft_edge_to_requested_first_layer": min_edge_to_first_layer,
        "design_skin_depth_um": float(delta * 1e6),
        "first_layer_per_design_skin_depth": float(
            delta / (profile["normal_um"] * 1e-6)
        ),
        "required_domains_present": all(d in domain_counts for d in range(1, 26)),
        "required_boundaries_present": all(
            d in set(map(int, np.unique(tagged.line_tags))) for d in range(1, 95)
        ),
        "quality_gate_pass": not rejection_reasons,
        "quality_gate_rejection_reasons": rejection_reasons,
    }
    (out / f"native_mqs_bl_{args.profile}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a native Gmsh skin-layer mesh for blocked MQS"
    )
    ap.add_argument(
        "--mphtxt",
        default=str(ROOT / "inputs/comsol_reference/Untitled.mphtxt"),
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="L2")
    ap.add_argument("--normal-um", type=float)
    ap.add_argument("--tangent-um", type=float)
    ap.add_argument("--thickness-um", type=float)
    ap.add_argument("--ratio", type=float)
    ap.add_argument("--global-h-mm", type=float, default=2.5)
    ap.add_argument("--fmax-Hz", type=float, default=8000.0)
    ap.add_argument("--sigma-S-m", type=float, default=1.12e7)
    ap.add_argument("--mu-r-design", type=float, default=1200.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quads", action="store_true")
    ap.add_argument(
        "--boundary-layer-curves",
        default="12,53,95,96,97,98",
        help="comma-separated COMSOL/Gmsh curve IDs",
    )
    ap.add_argument(
        "--corner-refinement-curves",
        default="",
        help="curves refined isotropically without a boundary-layer offset",
    )
    ap.add_argument("--corner-refinement-um", type=float, default=24.0)
    ap.add_argument(
        "--no-extend-from-boundary",
        action="store_true",
        help="do not propagate boundary point sizes through entire surfaces",
    )
    args = ap.parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
