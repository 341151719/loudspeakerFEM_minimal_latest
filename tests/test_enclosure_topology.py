from __future__ import annotations

from pathlib import Path
import resource
from time import perf_counter

import pytest

from loudspeaker_axisym_fem.enclosure_geometry import REFERENCE_CONFIGS, generate_reference_mesh
from loudspeaker_axisym_fem.enclosure_topology import audit_mesh


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"


def _critical_report_values(report: dict) -> dict:
    return {
        "geometry_signature": report["source"]["geometry_signature_sha256"],
        "mesh_counts": report["mesh_counts"],
        "quality": report["quality"],
        "domains": report["domains"],
        "interfaces": report["interfaces"],
        "connectivity": report["connectivity"],
        "traces": report["traces"],
        "checks": report["checks"],
    }


def test_all_reference_mesh_levels_pass_topology_and_converge(tmp_path):
    levels = ("L0", "L1", "L2")
    reports = {}
    generation_sizes = {}
    started = perf_counter()
    for case_id, config_name in sorted(REFERENCE_CONFIGS.items()):
        config_path = CONFIG_DIR / config_name
        reports[case_id] = {}
        generation_sizes[case_id] = {}
        for level in levels:
            mesh_path = tmp_path / f"{case_id}_{level}.msh"
            result = generate_reference_mesh(config_path, level, mesh_path)
            generation_sizes[case_id][level] = result.geometry["global_size_m"]
            reports[case_id][level] = audit_mesh(
                mesh_path,
                case_id=case_id,
                config_path=config_path,
            )

    for case_id, level_reports in reports.items():
        assert generation_sizes[case_id]["L0"] > generation_sizes[case_id]["L1"] > generation_sizes[case_id]["L2"]
        triangles = [level_reports[level]["mesh_counts"]["triangle_elements"] for level in levels]
        assert triangles[0] < triangles[1] < triangles[2], (case_id, triangles)
        for level, report in level_reports.items():
            assert report["status"] == "pass", (case_id, level, report["failures"])
            assert report["labels"]["final_production_interface_ready"] is False
            assert report["labels"]["reference_planar_piston"]["identity"] == "reference planar piston"
            assert report["volume_contract"]["air_cavity_volume_m3"] == pytest.approx(0.0061, rel=0.01)
            assert report["volume_contract"]["air_cavity_relative_error"] <= 0.01
            assert report["quality"]["minimum_triangle_quality"] >= 0.10
            assert report["quality"]["negative_r_point_count"] == 0
            assert report["quality"]["degenerate_triangle_count"] == 0
            assert report["interfaces"]["nonmanifold_edge_count"] == 0
            assert report["interfaces"]["uncovered_pressure_boundary_edge_count"] == 0
            assert report["interfaces"]["duplicate_line_edge_count"] == 0
            assert report["traces"]["reference_planar_piston"]["separated"] is True
            assert all(
                value["axisymmetric_volume_m3"] > 0.0
                for value in report["domains"]["rigid_displacement_domains"].values()
            )

        if case_id == "A":
            assert level_reports["L0"]["connectivity"]["cavity_to_exterior_path"] == [
                "air_cavity",
                "air_rear_opening",
                "air_rear_free",
            ]
        if case_id == "D":
            assert level_reports["L0"]["connectivity"]["cavity_to_exterior_path"] == [
                "air_cavity",
                "air_port",
                "air_rear_free",
            ]
        if case_id in ("B", "C", "E"):
            for level in levels:
                assert level_reports[level]["connectivity"]["cavity_to_exterior_path"] is None
                assert level_reports[level]["connectivity"]["cavity_pressure_neighbors"] == []

    assert reports["D"]["L0"]["volume_contract"]["air_port_volume_m3"] > 0.0
    assert reports["E"]["L0"]["traces"]["passive_radiator"]["separated"] is True
    for level in levels:
        assert _critical_report_values(reports["B"][level]) == _critical_report_values(reports["C"][level])

    elapsed = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"mesh_hierarchy_15 total_wall_s={elapsed:.3f} peak_rss_mb={peak_rss_mb:.1f}")
