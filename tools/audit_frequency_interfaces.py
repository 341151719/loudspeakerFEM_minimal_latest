#!/usr/bin/env python3
"""Audit low/mid-frequency acoustic, structure, NRA and Boundary93 geometry."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "best_model")]

from cli import _profile_for_frequency  # noqa: E402
from coupled_solver import load_config  # noqa: E402
from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio  # noqa: E402
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (  # noqa: E402
    ACOUSTIC_DOMAINS, NRA_DOMAINS, STRUCTURAL_DOMAINS,
    parse_mphtxt_boundary_adjacency,
)
from loudspeaker_axisym_fem.production_wet_trace import extract_production_wet_traces, sha256_file  # noqa: E402


def edges_by_triangle(mesh):
    edges = defaultdict(list)
    for index, triangle in enumerate(mesh.triangles):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges[tuple(sorted((int(a), int(b))))].append(index)
    return edges


def surface_areas(mesh, tags):
    result = {}
    for tag in tags:
        segments = mesh.line_cells[mesh.line_tags == tag]
        points = mesh.points_rz_m[segments]
        lengths = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
        result[tag] = float(np.sum(2 * math.pi * np.mean(points[:, :, 0], axis=1) * lengths))
    return result


def audit_profile(freq, config_path):
    cfg = load_config(ROOT, config_path)
    geometry = cfg["geometry"]
    acoustic_path = ROOT / geometry["mesh"]
    structure_path = ROOT / geometry.get("structure_mesh", geometry["mesh"])
    mphtxt_path = ROOT / geometry["mphtxt"]
    acoustic = load_tagged_meshio(acoustic_path)
    structure = acoustic if acoustic_path == structure_path else load_tagged_meshio(structure_path)
    adjacency = parse_mphtxt_boundary_adjacency(mphtxt_path)
    edge_triangles = edges_by_triangle(acoustic)
    centroids = acoustic.points_rz_m[acoustic.triangles].mean(axis=1)
    acoustic_counts = {str(domain): int(np.count_nonzero(acoustic.tri_domains == domain))
                       for domain in sorted(ACOUSTIC_DOMAINS)}
    wet_tags = sorted(tag for tag, item in adjacency.items()
                      if {item.up_domain, item.down_domain} & ACOUSTIC_DOMAINS
                      and {item.up_domain, item.down_domain} & STRUCTURAL_DOMAINS)
    wet_acoustic = surface_areas(acoustic, wet_tags)
    wet_structure = surface_areas(structure, wet_tags)
    failures = []
    if any(count == 0 for count in acoustic_counts.values()):
        failures.append("one or more required acoustic domains are absent")
    acoustic_triangles = acoustic.points_rz_m[acoustic.triangles[np.isin(acoustic.tri_domains, list(ACOUSTIC_DOMAINS))]]
    first_ac = acoustic_triangles[:, 1] - acoustic_triangles[:, 0]
    second_ac = acoustic_triangles[:, 2] - acoustic_triangles[:, 0]
    acoustic_double_area = first_ac[:, 0] * second_ac[:, 1] - first_ac[:, 1] * second_ac[:, 0]
    if np.any(np.abs(acoustic_double_area) <= 0):
        failures.append("acoustic mesh contains a zero-area triangle")
    max_area_error = 0.0
    for tag in wet_tags:
        a, b = wet_acoustic[tag], wet_structure[tag]
        if a <= 0 or b <= 0:
            failures.append(f"wet boundary {tag} missing or zero area")
        elif abs(a - b) / a > 0.005:
            failures.append(f"wet boundary {tag} area mismatch exceeds 0.5%")
        if a > 0:
            max_area_error = max(max_area_error, abs(a - b) / a)

    wet_segments = 0
    boundary93_dots = []
    boundary93_radii = []
    nra_boundary_segments = {domain: 0 for domain in NRA_DOMAINS}
    for segment, tag_raw in zip(acoustic.line_cells, acoustic.line_tags):
        tag = int(tag_raw)
        item = adjacency.get(tag)
        if item is None:
            continue
        expected = {item.up_domain, item.down_domain}
        is_nra_boundary = bool(expected & NRA_DOMAINS.keys())
        if tag not in wet_tags and tag != 93 and not is_nra_boundary:
            continue
        adjacent = edge_triangles.get(tuple(sorted(map(int, segment))), [])
        domains = {int(acoustic.tri_domains[index]) for index in adjacent}
        if domains != (expected - {0}):
            failures.append(f"boundary {tag} edge has domains {sorted(domains)}, expected {sorted(expected)}")
            continue
        points = acoustic.points_rz_m[segment]
        tangent = points[1] - points[0]
        length = float(np.linalg.norm(tangent))
        if length <= 0:
            failures.append(f"boundary {tag} has zero-length edge")
            continue
        normal = np.array([tangent[1], -tangent[0]]) / length
        if tag in wet_tags:
            wet_segments += 1
            adom = next(iter(expected & ACOUSTIC_DOMAINS))
            sdom = next(iter(expected & STRUCTURAL_DOMAINS))
            ac_tri = next(index for index in adjacent if int(acoustic.tri_domains[index]) == adom)
            st_tri = next(index for index in adjacent if int(acoustic.tri_domains[index]) == sdom)
            if normal @ (centroids[st_tri] - centroids[ac_tri]) < 0:
                normal = -normal
            if normal @ (centroids[st_tri] - centroids[ac_tri]) <= 0:
                failures.append(f"wet boundary {tag} normal does not point acoustic-to-structure")
        elif tag == 93:
            if expected != {4, 5}:
                failures.append("Boundary93 must separate physical acoustic domain 4 and PML domain 5")
                continue
            physical = next(index for index in adjacent if int(acoustic.tri_domains[index]) == 4)
            pml = next(index for index in adjacent if int(acoustic.tri_domains[index]) == 5)
            if normal @ (centroids[pml] - centroids[physical]) < 0:
                normal = -normal
            midpoint = points.mean(axis=0)
            radius = float(np.linalg.norm(midpoint))
            boundary93_radii.append(radius)
            boundary93_dots.append(float(normal @ midpoint / radius))
        else:
            for domain in expected & NRA_DOMAINS.keys():
                nra_boundary_segments[domain] += 1
    if not boundary93_dots or min(boundary93_dots) < 0.99:
        failures.append("Boundary93 radial normal alignment below 0.99 or boundary missing")

    nra = {}
    for domain, height in NRA_DOMAINS.items():
        triangles = acoustic.triangles[acoustic.tri_domains == domain]
        count = len(triangles)
        points = acoustic.points_rz_m[triangles]
        first = points[:, 1] - points[:, 0]
        second = points[:, 2] - points[:, 0]
        signed_double_areas = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        min_area = float(np.min(np.abs(signed_double_areas)) / 2) if count else 0.0
        configured = float(cfg["nra"]["domains"][str(domain)]["height_m"])
        nra[str(domain)] = {"triangles": count, "model_height_m": height,
                            "config_height_m": configured,
                            "min_triangle_area_m2": min_area,
                            "tagged_boundary_segments": nra_boundary_segments[domain]}
        if count == 0 or min_area <= 0 or nra_boundary_segments[domain] == 0 or not math.isclose(height, configured, rel_tol=1e-12):
            failures.append(f"NRA domain {domain} geometry or height mismatch")

    wet_trace = extract_production_wet_traces(acoustic_path, mphtxt_path=mphtxt_path)
    if wet_trace["status"] != "pass":
        failures.append("production wet-trace topology audit failed")
    return {
        "frequency_Hz": float(freq), "config": str(config_path or "configs/best_model.json"),
        "acoustic_mesh": str(acoustic_path.relative_to(ROOT)), "acoustic_mesh_sha256": sha256_file(acoustic_path),
        "structure_mesh": str(structure_path.relative_to(ROOT)), "structure_mesh_sha256": sha256_file(structure_path),
        "mphtxt_sha256": sha256_file(mphtxt_path),
        "acoustic_domains": acoustic_counts,
        "min_acoustic_triangle_area_m2": float(np.min(np.abs(acoustic_double_area)) / 2),
        "nra": nra,
        "wet_interface": {"boundary_ids": wet_tags, "segments": wet_segments,
                          "max_structure_area_relative_error": max_area_error,
                          "front_ids": wet_trace["front"]["boundary_ids"],
                          "rear_ids": wet_trace["rear"]["boundary_ids"]},
        "boundary93": {"segments": len(boundary93_dots),
                       "min_normal_radial_dot": min(boundary93_dots) if boundary93_dots else None,
                       "radius_min_m": min(boundary93_radii) if boundary93_radii else None,
                       "radius_max_m": max(boundary93_radii) if boundary93_radii else None},
        "reference_planar_piston_mismatch": wet_trace["comparison"]["mismatch"],
        "failures": sorted(set(failures)), "status": "pass" if not failures else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freqs", type=float, nargs="+", default=[50.0, 6300.0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, nargs="*", default=[],
                        help="optional solved summary JSON files for coupling and Boundary93 closure")
    args = parser.parse_args()
    base = load_config(ROOT)
    hybrid = base["acoustics"]["hybrid_sweep"]
    profiles = [audit_profile(freq, _profile_for_frequency(None, hybrid, freq)) for freq in args.freqs]
    summaries = {float(data["freq_Hz"]): data for data in (
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    )}
    for profile in profiles:
        data = summaries.get(profile["frequency_Hz"])
        if data is None:
            continue
        coupling = data["metadata"]["G"]
        boundary93 = data["metadata"]["boundary93"]
        pml = data["pml"]
        profile["solved_coupling"] = {
            "G_shape": coupling["G_shape"],
            "missed_boundary_tags": coupling["missed_boundary_tags"],
            "interface_boundaries": coupling["interface_boundaries"],
            "max_projection_distance_m": coupling.get("max_projection_distance_m"),
            "nra_enabled": data["metadata"]["nra_enabled"],
            "boundary93_source_segments": boundary93["source_segments"],
            "boundary93_force_radial_normals": boundary93["force_radial_normals"],
            "nra_model": pml.get("nra_model"),
            "nra_factors": pml.get("nra_factors"),
        }
        if (coupling["missed_boundary_tags"] or
            coupling["interface_boundaries"] != profile["wet_interface"]["boundary_ids"] or
            boundary93["source_segments"] != profile["boundary93"]["segments"] or
            not boundary93["force_radial_normals"] or not data["metadata"]["nra_enabled"] or
            pml.get("nra_model") != "native_parallel_plate_thermoviscous" or
            set(pml.get("nra_factors", {})) != {"8", "22"}):
            profile["failures"].append("solved coupling/Boundary93/NRA metadata differs from geometry audit")
            profile["status"] = "fail"
    report = {"profiles": profiles, "status": "pass" if all(p["status"] == "pass" for p in profiles) else "fail"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "profiles": [
        {"frequency_Hz": p["frequency_Hz"], "status": p["status"], "failures": p["failures"],
         "wet_area_error": p["wet_interface"]["max_structure_area_relative_error"],
         "boundary93_min_dot": p["boundary93"]["min_normal_radial_dot"]} for p in profiles]}, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
