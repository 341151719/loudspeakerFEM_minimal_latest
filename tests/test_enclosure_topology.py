from __future__ import annotations

from pathlib import Path

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


def test_all_l0_reference_meshes_pass_topology_audit_and_volume_contract(tmp_path):
    reports = {}
    for case_id, config_name in sorted(REFERENCE_CONFIGS.items()):
        config_path = CONFIG_DIR / config_name
        mesh_path = tmp_path / f"{case_id}.msh"
        generate_reference_mesh(config_path, "L0", mesh_path)
        reports[case_id] = audit_mesh(mesh_path, case_id=case_id, config_path=config_path)

    for case_id, report in reports.items():
        assert report["status"] == "pass", (case_id, report["failures"])
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

    assert reports["A"]["connectivity"]["cavity_to_exterior_path"] == [
        "air_cavity",
        "air_rear_opening",
        "air_rear_free",
    ]
    assert reports["D"]["connectivity"]["cavity_to_exterior_path"] == [
        "air_cavity",
        "air_port",
        "air_rear_free",
    ]
    for case_id in ("B", "C", "E"):
        assert reports[case_id]["connectivity"]["cavity_to_exterior_path"] is None
        assert reports[case_id]["connectivity"]["cavity_pressure_neighbors"] == []

    assert reports["D"]["volume_contract"]["air_port_volume_m3"] > 0.0
    assert reports["E"]["traces"]["passive_radiator"]["separated"] is True
    assert _critical_report_values(reports["B"]) == _critical_report_values(reports["C"])
