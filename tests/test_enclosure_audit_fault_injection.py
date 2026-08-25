from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import meshio
import numpy as np
import pytest

from loudspeaker_axisym_fem.enclosure_geometry import generate_reference_mesh
from loudspeaker_axisym_fem.enclosure_topology import audit_mesh
from loudspeaker_axisym_fem.production_wet_trace import (
    DEFAULT_PRODUCTION_MESH,
    extract_production_wet_traces,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "enclosures"
CASE_CONFIGS = {
    "A": "open_back.json",
    "D": "vented_rear_coaxial.json",
}


def _reference_mesh(tmp_path: Path, case_id: str) -> Path:
    config_name = {
        "A": "open_back.json",
        "B": "sealed_lossless.json",
        "D": "vented_rear_coaxial.json",
    }[case_id]
    path = tmp_path / f"{case_id}_L0.msh"
    generate_reference_mesh(CONFIG_DIR / config_name, "L0", path)
    return path


def _tag(mesh: meshio.Mesh, name: str) -> int:
    return int(np.asarray(mesh.field_data[name]).reshape(-1)[0])


def _write_variant(
    source: Path,
    destination: Path,
    *,
    field_mutator=None,
    line_mutator=None,
    triangle_mutator=None,
) -> None:
    mesh = meshio.read(source)
    lines = np.array(mesh.cells_dict["line"], dtype=np.int64, copy=True)
    triangles = np.array(mesh.cells_dict["triangle"], dtype=np.int64, copy=True)
    line_tags = np.array(mesh.cell_data_dict["gmsh:physical"]["line"], dtype=np.int64, copy=True)
    triangle_tags = np.array(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int64, copy=True)
    line_geometrical = np.array(mesh.cell_data_dict["gmsh:geometrical"]["line"], dtype=np.int64, copy=True)
    triangle_geometrical = np.array(mesh.cell_data_dict["gmsh:geometrical"]["triangle"], dtype=np.int64, copy=True)
    field_data = {
        name: np.array(value, dtype=np.int64, copy=True)
        for name, value in mesh.field_data.items()
    }
    if field_mutator is not None:
        field_mutator(field_data)
    if line_mutator is not None:
        lines, line_tags = line_mutator(lines, line_tags)
    if triangle_mutator is not None:
        triangle_tags = triangle_mutator(triangles, triangle_tags, field_data)
    if len(line_geometrical) < len(lines):
        line_geometrical = np.concatenate(
            [
                line_geometrical,
                np.full(
                    len(lines) - len(line_geometrical),
                    int(line_geometrical[0]),
                    dtype=np.int64,
                ),
            ]
        )
    point_data = {
        name: np.array(value, copy=True)
        for name, value in mesh.point_data.items()
    }
    if "gmsh:dim_tags" not in point_data:
        point_data["gmsh:dim_tags"] = np.column_stack(
            [
                np.zeros(len(mesh.points), dtype=np.int64),
                np.ones(len(mesh.points), dtype=np.int64),
            ]
        )
    variant = meshio.Mesh(
        points=np.array(mesh.points, dtype=float, copy=True),
        cells=[("line", lines), ("triangle", triangles)],
        cell_data={
            "gmsh:physical": [line_tags, triangle_tags],
            "gmsh:geometrical": [line_geometrical[: len(lines)], triangle_geometrical],
        },
        point_data=point_data,
        field_data=field_data,
    )
    # MSH 2.2 preserves the supplied physical/geometrical arrays without
    # requiring Gmsh 4 entity blocks, which keeps the fault mutation readable.
    meshio.write(destination, variant, file_format="gmsh22")


def _triangle_with_neighbor(triangles: np.ndarray, tags: np.ndarray, left: int, right: int) -> int:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, triangle in enumerate(triangles):
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((first, second)))].append(index)
    for indices in edges.values():
        if any(int(tags[index]) == left for index in indices) and any(int(tags[index]) == right for index in indices):
            return next(index for index in indices if int(tags[index]) == left)
    raise AssertionError(f"no triangle adjacency {left} <-> {right}")


def test_fault_injected_wrong_air_cavity_tag_fails_exact_physical_contract(tmp_path):
    source = _reference_mesh(tmp_path, "B")
    variant = tmp_path / "wrong_cavity_tag.msh"

    def mutate(fields):
        fields["air_cavity"][0] = 9999

    _write_variant(source, variant, field_mutator=mutate)
    report = audit_mesh(variant, case_id="B", config_path=CONFIG_DIR / "sealed_lossless.json")
    assert report["status"] == "fail"
    assert "physical_group_contract_exact" in report["failures"]
    mismatches = report["group_contract"]["physical_group_contract"]["mismatches"]
    assert any(row["name"] == "air_cavity" for row in mismatches)


def test_fault_injected_expected_group_dimension_fails_exact_physical_contract(tmp_path):
    source = _reference_mesh(tmp_path, "B")
    variant = tmp_path / "wrong_cavity_dimension.msh"

    def mutate(fields):
        fields["air_cavity"][1] = 1

    _write_variant(source, variant, field_mutator=mutate)
    report = audit_mesh(variant, case_id="B", config_path=CONFIG_DIR / "sealed_lossless.json")
    contract = report["group_contract"]["physical_group_contract"]
    assert report["status"] == "fail"
    assert "physical_group_contract_exact" in report["failures"]
    assert any(row["name"] == "air_cavity" for row in contract["mismatches"])


def test_fault_injected_duplicate_same_dimension_tag_fails_exact_physical_contract(tmp_path):
    source = _reference_mesh(tmp_path, "B")
    variant = tmp_path / "duplicate_domain_tag.msh"

    def mutate(fields):
        fields["air_cavity_alias"] = np.array(
            [int(fields["air_cavity"][0]), 2], dtype=np.int64
        )

    _write_variant(source, variant, field_mutator=mutate)
    report = audit_mesh(variant, case_id="B", config_path=CONFIG_DIR / "sealed_lossless.json")
    contract = report["group_contract"]["physical_group_contract"]
    assert report["status"] == "fail"
    assert "physical_group_contract_exact" in report["failures"]
    assert any(
        any(row["name"] == "air_cavity_alias" for row in rows)
        for rows in contract["duplicate_tags"].values()
    )


@pytest.mark.parametrize("case_id", ["A", "D"])
def test_fault_injected_bridge_bypass_fails_exact_pressure_adjacency(tmp_path, case_id):
    source = _reference_mesh(tmp_path, case_id)
    variant = tmp_path / f"{case_id}_bridge_bypass.msh"
    mesh = meshio.read(source)
    side_tag = _tag(mesh, "air_side_free")
    rear_tag = _tag(mesh, "air_rear_free")
    bridge_name = "air_rear_opening" if case_id == "A" else "air_port"
    bridge_tag = _tag(mesh, bridge_name)

    def mutate(triangles, triangle_tags, _fields):
        index = _triangle_with_neighbor(triangles, triangle_tags, rear_tag, side_tag)
        triangle_tags[index] = bridge_tag
        return triangle_tags

    _write_variant(source, variant, triangle_mutator=mutate)
    report = audit_mesh(variant, case_id=case_id, config_path=CONFIG_DIR / CASE_CONFIGS[case_id])
    assert report["status"] == "fail"
    assert "exact_pressure_pressure_adjacency_contract" in report["failures"]
    pressure_contract = report["adjacency_contract"]["pressure_pressure"]
    assert pressure_contract["missing"] or pressure_contract["unexpected"]


def test_fault_injected_production_wet_trace_missing_line_owner_fails(tmp_path):
    source = ROOT / DEFAULT_PRODUCTION_MESH
    variant = tmp_path / "wet_trace_missing_owner.msh"

    def mutate(lines, tags):
        keep = np.flatnonzero(tags != 47)
        return lines[keep], tags[keep]

    _write_variant(source, variant, line_mutator=mutate)
    report = extract_production_wet_traces(variant, mphtxt_path=None)
    assert report["status"] == "fail"
    assert "target_common_edge_line_owner" in report["failures"]
    assert report["integrity"]["missing_line_owners"]
    assert report["integrity"]["missing_common_edges"]


def test_fault_injected_production_wet_trace_duplicate_owner_fails(tmp_path):
    source = ROOT / DEFAULT_PRODUCTION_MESH
    variant = tmp_path / "wet_trace_duplicate_owner.msh"

    def mutate(lines, tags):
        index = int(np.flatnonzero(tags == 47)[0])
        return np.concatenate([lines, lines[index : index + 1]]), np.concatenate([tags, tags[index : index + 1]])

    _write_variant(source, variant, line_mutator=mutate)
    report = extract_production_wet_traces(variant, mphtxt_path=None)
    assert report["status"] == "fail"
    assert "target_common_edge_line_owner" in report["failures"]
    assert report["integrity"]["duplicate_line_owners"]


def test_fault_injected_production_wet_trace_missing_structural_partition_fails(tmp_path):
    source = ROOT / DEFAULT_PRODUCTION_MESH
    variant = tmp_path / "wet_trace_missing_structural_partition.msh"

    def mutate(triangles, triangle_tags, fields):
        triangle_tags[triangle_tags == int(fields["domain_25"][0])] = int(fields["domain_21"][0])
        return triangle_tags

    _write_variant(source, variant, triangle_mutator=mutate)
    report = extract_production_wet_traces(variant, mphtxt_path=None)
    assert report["status"] == "fail"
    assert "target_structural_partition_coverage" in report["failures"]
    assert 25 in report["integrity"]["missing_front_structural_domains"]
    assert 25 in report["integrity"]["missing_rear_structural_domains"]
