"""Pure meshio topology and physics-label audit for the enclosure L0 meshes.

This module does not assemble or solve a FEM problem.  It treats the Gmsh
physical 2-D groups named ``air_*`` as pressure domains and ``rigid_*`` as
rigid displacement domains, then checks their mesh adjacency and rotated
axisymmetric measures.  The reference planar piston is deliberately kept a
named boundary trace; this audit never treats it as a production interface.
"""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import meshio
import numpy as np

from .enclosure_geometry import (
    BOUNDARY_PHYSICAL_TAGS,
    DOMAIN_PHYSICAL_TAGS,
    REFERENCE_CONFIGS,
    case_id_for_config,
    expected_boundary_names,
    expected_domain_names,
)
from .enclosure_schema import load_enclosure_config


REFERENCE_NET_VOLUME_TARGET_M3 = 0.0061
AREA_TOLERANCE_M2 = 1.0e-15
NEGATIVE_R_TOLERANCE_M = 1.0e-12
MIN_TRIANGLE_QUALITY = 0.10
VOLUME_RELATIVE_TOLERANCE = 0.01

PRESSURE_DOMAIN_PREFIX = "air_"
RIGID_DOMAIN_PREFIX = "rigid_"
REFERENCE_PLANAR_PISTON_FRONT = "reference_planar_piston_front"
REFERENCE_PLANAR_PISTON_BACK = "reference_planar_piston_back"
PR_CAVITY_FACE = "pr_cavity_face"
PR_EXTERIOR_FACE = "pr_exterior_face"


class MeshTopologyAuditError(ValueError):
    """Raised when a mesh cannot be read as a tagged triangular mesh."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(digest: "hashlib._Hash", value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _geometry_signature(mesh: meshio.Mesh) -> str:
    """Hash geometry, cell connectivity, physical tags, and named groups."""

    digest = hashlib.sha256()
    _array_digest(digest, np.asarray(mesh.points))
    for cell_type in ("line", "triangle"):
        blocks = [block.data for block in mesh.cells if block.type == cell_type]
        if blocks:
            _array_digest(digest, np.concatenate(blocks, axis=0))
        else:
            _array_digest(digest, np.empty((0, 2 if cell_type == "line" else 3), dtype=np.int64))
        physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
        if physical is None:
            _array_digest(digest, np.empty((0,), dtype=np.int64))
        else:
            _array_digest(digest, np.asarray(physical, dtype=np.int64))
    for name in sorted(mesh.field_data):
        digest.update(name.encode("utf-8"))
        _array_digest(digest, np.asarray(mesh.field_data[name], dtype=np.int64))
    return digest.hexdigest()


def _concat_cells(mesh: meshio.Mesh, cell_type: str, width: int) -> np.ndarray:
    blocks = [np.asarray(block.data, dtype=np.int64) for block in mesh.cells if block.type == cell_type]
    if not blocks:
        return np.empty((0, width), dtype=np.int64)
    if any(block.ndim != 2 or block.shape[1] != width for block in blocks):
        raise MeshTopologyAuditError(f"{cell_type} cells must have shape (n, {width})")
    return np.concatenate(blocks, axis=0)


def _physical_tags(mesh: meshio.Mesh, cell_type: str, count: int) -> np.ndarray:
    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    if physical is None:
        raise MeshTopologyAuditError(f"mesh has no gmsh:physical tags for {cell_type} cells")
    tags = np.asarray(physical, dtype=np.int64).reshape(-1)
    if len(tags) != count:
        raise MeshTopologyAuditError(
            f"{cell_type} physical-tag count {len(tags)} does not match cell count {count}"
        )
    return tags


def _field_maps(mesh: meshio.Mesh) -> tuple[dict[int, str], dict[int, str], dict[str, list[int]]]:
    by_dimension: dict[int, dict[int, str]] = {1: {}, 2: {}}
    duplicate_tags: dict[str, list[int]] = defaultdict(list)
    for name in sorted(mesh.field_data):
        value = np.asarray(mesh.field_data[name]).reshape(-1)
        if len(value) < 2:
            raise MeshTopologyAuditError(f"physical group {name!r} has malformed field_data")
        tag, dimension = int(value[0]), int(value[1])
        if dimension not in by_dimension:
            continue
        old = by_dimension[dimension].get(tag)
        if old is not None and old != name:
            duplicate_tags[str(dimension)].extend([tag])
        by_dimension[dimension][tag] = name
    return by_dimension[1], by_dimension[2], duplicate_tags


def _edge_key(edge: Iterable[int]) -> tuple[int, int]:
    a, b = (int(value) for value in edge)
    return (a, b) if a < b else (b, a)


def _triangle_edges(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, triangle in enumerate(triangles):
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_to_triangles[_edge_key(edge)].append(int(index))
    return dict(edge_to_triangles)


def _line_edges(lines: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_to_lines: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        edge_to_lines[_edge_key(line)].append(int(index))
    return dict(edge_to_lines)


def _components_for_domain(
    triangle_indices: np.ndarray,
    edge_to_triangles: Mapping[tuple[int, int], list[int]],
    domain_tags: np.ndarray,
    tag: int,
) -> list[list[int]]:
    members = {int(index) for index in triangle_indices}
    adjacency: dict[int, set[int]] = {index: set() for index in members}
    for indices in edge_to_triangles.values():
        same = [index for index in indices if index in members and int(domain_tags[index]) == int(tag)]
        for left in same:
            adjacency[left].update(right for right in same if right != left)
    components: list[list[int]] = []
    remaining = set(members)
    while remaining:
        start = min(remaining)
        queue = [start]
        remaining.remove(start)
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _graph_components(nodes: Iterable[str], edges: Mapping[tuple[str, str], int]) -> list[dict[str, Any]]:
    node_set = set(nodes)
    adjacency: dict[str, set[str]] = {node: set() for node in node_set}
    for (left, right), count in edges.items():
        if count <= 0:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    remaining = set(node_set)
    output: list[dict[str, Any]] = []
    while remaining:
        start = min(remaining)
        queue = [start]
        remaining.remove(start)
        component: set[str] = set()
        while queue:
            current = queue.pop()
            component.add(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        component_edges = sum(
            count
            for (left, right), count in edges.items()
            if left in component and right in component
        )
        output.append(
            {
                "domains": sorted(component),
                "edge_count": int(component_edges),
            }
        )
    return sorted(output, key=lambda item: item["domains"])


def _shortest_path(
    graph_edges: Mapping[tuple[str, str], int],
    start: str,
    targets: Iterable[str],
) -> list[str] | None:
    target_set = set(targets)
    if start in target_set:
        return [start]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in graph_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    queue: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}
    found: str | None = None
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor in target_set:
                found = neighbor
                queue.clear()
                break
            queue.append(neighbor)
    if found is None:
        return None
    path: list[str] = []
    current: str | None = found
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _pair_rows(pair_counts: Mapping[tuple[str, str], int]) -> list[dict[str, Any]]:
    return [
        {"domains": [left, right], "edge_count": int(pair_counts[(left, right)])}
        for left, right in sorted(pair_counts)
    ]


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(int(value))] += 1
    return {key: counts[key] for key in sorted(counts, key=lambda item: int(item))}


def _case_from_mesh(mesh: meshio.Mesh, requested: str | None) -> str:
    if requested is not None:
        case_id = str(requested).upper()
        if case_id not in REFERENCE_CONFIGS:
            raise MeshTopologyAuditError(f"unknown reference case {requested!r}")
        return case_id
    names = set(mesh.field_data)
    if "air_rear_opening" in names:
        return "A"
    if "air_port" in names:
        return "D"
    if "rigid_pr_back_mechanism" in names:
        return "E"
    return "B"


def _config_details(config_path: str | Path | None, case_id: str) -> dict[str, Any]:
    if config_path is None:
        return {
            "path": None,
            "sha256": None,
            "net_volume_target_m3": REFERENCE_NET_VOLUME_TARGET_M3,
            "case": None,
        }
    path = Path(config_path)
    config = load_enclosure_config(path)
    actual_case = case_id_for_config(config.case)
    if actual_case != case_id:
        raise MeshTopologyAuditError(
            f"config case {config.case!r} maps to {actual_case}, not requested {case_id}"
        )
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "net_volume_target_m3": float(config.net_volume_target_m3),
        "case": config.case,
    }


def audit_mesh(
    mesh_path: str | Path,
    *,
    case_id: str | None = None,
    config_path: str | Path | None = None,
    volume_relative_tolerance: float = VOLUME_RELATIVE_TOLERANCE,
    minimum_triangle_quality: float = MIN_TRIANGLE_QUALITY,
) -> dict[str, Any]:
    """Audit one tagged triangular reference mesh and return JSON-safe data."""

    path = Path(mesh_path)
    try:
        mesh = meshio.read(path)
    except Exception as exc:  # pragma: no cover - meshio supplies the detail
        raise MeshTopologyAuditError(f"cannot read mesh {path}: {exc}") from exc
    if len(mesh.points) == 0:
        raise MeshTopologyAuditError("mesh has no points")
    if mesh.points.shape[1] < 2:
        raise MeshTopologyAuditError("axisymmetric mesh needs r,z point coordinates")

    lines = _concat_cells(mesh, "line", 2)
    triangles = _concat_cells(mesh, "triangle", 3)
    triangle_tags = _physical_tags(mesh, "triangle", len(triangles))
    line_tags = _physical_tags(mesh, "line", len(lines)) if len(lines) else np.empty((0,), dtype=np.int64)
    line_names, domain_names_by_tag, duplicate_tags = _field_maps(mesh)
    case = _case_from_mesh(mesh, case_id)
    config = _config_details(config_path, case)

    points = np.asarray(mesh.points[:, :2], dtype=float)
    triangles_rz = points[triangles]
    edge_a = triangles_rz[:, 1] - triangles_rz[:, 0]
    edge_b = triangles_rz[:, 2] - triangles_rz[:, 0]
    signed_cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    areas = np.abs(signed_cross) * 0.5
    centroids = np.mean(triangles_rz, axis=1)
    rotated_volumes = 2.0 * math.pi * centroids[:, 0] * areas
    edge_lengths_squared = np.sum((triangles_rz - np.roll(triangles_rz, -1, axis=1)) ** 2, axis=2)
    denominator = np.sum(edge_lengths_squared, axis=1)
    qualities = np.divide(
        4.0 * math.sqrt(3.0) * areas,
        denominator,
        out=np.zeros_like(areas),
        where=denominator > 0.0,
    )

    edge_to_triangles = _triangle_edges(triangles)
    edge_to_lines = _line_edges(lines)
    triangle_edge_adjacencies = [len(indices) for indices in edge_to_triangles.values()]
    nonmanifold_edges = sorted(
        edge for edge, indices in edge_to_triangles.items() if len(indices) > 2
    )
    single_triangle_edges = {
        edge for edge, indices in edge_to_triangles.items() if len(indices) == 1
    }
    line_edges = set(edge_to_lines)
    uncovered_boundary_edges = sorted(single_triangle_edges - line_edges)
    duplicate_line_edges = sorted(edge for edge, indices in edge_to_lines.items() if len(indices) > 1)

    pressure_names = sorted(
        name for name in domain_names_by_tag.values() if name.startswith(PRESSURE_DOMAIN_PREFIX)
    )
    rigid_names = sorted(
        name for name in domain_names_by_tag.values() if name.startswith(RIGID_DOMAIN_PREFIX)
    )
    expected_domains = set(expected_domain_names(case))
    expected_boundaries = set(expected_boundary_names(case))
    actual_domains = set(domain_names_by_tag.values())
    actual_boundaries = set(line_names.values())
    missing_domains = sorted(expected_domains - actual_domains)
    missing_boundaries = sorted(expected_boundaries - actual_boundaries)
    unexpected_domains = sorted(actual_domains - expected_domains)
    unexpected_boundaries = sorted(actual_boundaries - expected_boundaries)

    tag_to_name = domain_names_by_tag
    unknown_triangle_tags = sorted(int(tag) for tag in set(triangle_tags) if int(tag) not in tag_to_name)
    unknown_line_tags = sorted(int(tag) for tag in set(line_tags) if int(tag) not in line_names)

    domain_stats: dict[str, dict[str, Any]] = {}
    all_domain_tags = sorted(set(int(tag) for tag in triangle_tags))
    for tag in all_domain_tags:
        name = tag_to_name.get(tag, f"physical_2d_tag_{tag}")
        mask = triangle_tags == tag
        indices = np.flatnonzero(mask)
        components = _components_for_domain(indices, edge_to_triangles, triangle_tags, tag)
        domain_stats[name] = {
            "physical_tag": int(tag),
            "physical_dimension": 2,
            "triangle_count": int(np.count_nonzero(mask)),
            "component_count": int(len(components)),
            "component_triangle_counts": [int(len(component)) for component in components],
            "area_m2": float(np.sum(areas[mask])),
            "axisymmetric_volume_m3": float(np.sum(rotated_volumes[mask])),
            "minimum_triangle_area_m2": float(np.min(areas[mask])) if np.any(mask) else 0.0,
            "minimum_triangle_quality": float(np.min(qualities[mask])) if np.any(mask) else 0.0,
            "r_range_m": [float(np.min(centroids[mask, 0])), float(np.max(centroids[mask, 0]))]
            if np.any(mask)
            else [None, None],
            "z_range_m": [float(np.min(centroids[mask, 1])), float(np.max(centroids[mask, 1]))]
            if np.any(mask)
            else [None, None],
        }

    pressure_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    pressure_rigid_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    same_domain_edge_count = 0
    for indices in edge_to_triangles.values():
        if len(indices) != 2:
            continue
        left = tag_to_name.get(int(triangle_tags[indices[0]]), f"physical_2d_tag_{int(triangle_tags[indices[0]])}")
        right = tag_to_name.get(int(triangle_tags[indices[1]]), f"physical_2d_tag_{int(triangle_tags[indices[1]])}")
        if left == right:
            same_domain_edge_count += 1
        elif left in pressure_names and right in pressure_names:
            pressure_pair_counts[_pair_key(left, right)] += 1
        elif (left in pressure_names and right in rigid_names) or (left in rigid_names and right in pressure_names):
            pressure_rigid_pair_counts[_pair_key(left, right)] += 1

    pressure_graph_components = _graph_components(pressure_names, pressure_pair_counts)
    cavity = "air_cavity"
    bridge_domains = {"air_rear_opening", "air_port"}
    exterior_pressure = sorted(
        name for name in pressure_names if name != cavity and name not in bridge_domains
    )
    cavity_neighbors = sorted(
        neighbor
        for pair in pressure_pair_counts
        if cavity in pair
        for neighbor in pair
        if neighbor != cavity
    )
    bridge_domain = {"A": "air_rear_opening", "D": "air_port"}.get(case)
    cavity_path_targets = ["air_rear_free"] if bridge_domain else exterior_pressure
    cavity_path = (
        _shortest_path(pressure_pair_counts, cavity, cavity_path_targets)
        if cavity in pressure_names
        else None
    )
    external_edges = {
        pair: count
        for pair, count in pressure_pair_counts.items()
        if pair[0] in exterior_pressure and pair[1] in exterior_pressure
    }
    exterior_components = _graph_components(exterior_pressure, external_edges)
    external_connected = len(exterior_components) == 1 and set(exterior_components[0]["domains"]) == set(exterior_pressure)
    allowed_cavity_neighbors = [bridge_domain] if bridge_domain else []
    cavity_can_reach_exterior = cavity_path is not None
    if case in {"A", "D"}:
        required_path = [cavity, bridge_domain, "air_rear_free"] if bridge_domain else []
        required_connectivity = (
            cavity_can_reach_exterior
            and cavity_neighbors == allowed_cavity_neighbors
            and cavity_path == required_path
        )
    else:
        required_path = []
        required_connectivity = not cavity_can_reach_exterior and not cavity_neighbors
    pressure_connectivity = {
        "domain_graph_edges": _pair_rows(pressure_pair_counts),
        "connected_components": pressure_graph_components,
        "external_pressure_domains": exterior_pressure,
        "external_connected": bool(external_connected),
        "cavity_pressure_neighbors": cavity_neighbors,
        "cavity_to_exterior_path": cavity_path,
        "expected_cavity_to_exterior_path": required_path or None,
        "allowed_cavity_pressure_neighbors": allowed_cavity_neighbors,
        "cavity_can_reach_exterior": bool(cavity_can_reach_exterior),
        "required_case_connectivity": bool(required_connectivity and external_connected),
    }

    line_group_stats: dict[str, dict[str, Any]] = {}
    line_group_edges: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for index, (line, tag) in enumerate(zip(lines, line_tags)):
        name = line_names.get(int(tag), f"physical_1d_tag_{int(tag)}")
        edge = _edge_key(line)
        line_group_edges[name].add(edge)
        adjacent = edge_to_triangles.get(edge, [])
        adjacent_domains = sorted(
            {
                tag_to_name.get(int(triangle_tags[triangle]), f"physical_2d_tag_{int(triangle_tags[triangle])}")
                for triangle in adjacent
            }
        )
        row = line_group_stats.setdefault(
            name,
            {
                "physical_tag": int(tag),
                "physical_dimension": 1,
                "line_count": 0,
                "unique_geometric_edge_count": 0,
                "unmatched_line_count": 0,
                "line_adjacency_histogram": defaultdict(int),
                "adjacent_domain_pairs": defaultdict(int),
            },
        )
        row["line_count"] += 1
        row["line_adjacency_histogram"][str(len(adjacent))] += 1
        if len(adjacent) == 0:
            row["unmatched_line_count"] += 1
        pair = tuple(adjacent_domains)
        row["adjacent_domain_pairs"][json.dumps(pair, ensure_ascii=False)] += 1
        _ = index

    for name, row in line_group_stats.items():
        row["unique_geometric_edge_count"] = len(line_group_edges[name])
        row["duplicate_geometric_edge_count"] = int(row["line_count"] - row["unique_geometric_edge_count"])
        row["line_adjacency_histogram"] = {
            key: int(value)
            for key, value in sorted(row["line_adjacency_histogram"].items(), key=lambda item: int(item[0]))
        }
        pairs: list[dict[str, Any]] = []
        for encoded, count in sorted(row["adjacent_domain_pairs"].items()):
            pairs.append({"domains": list(json.loads(encoded)), "line_count": int(count)})
        row["adjacent_domain_pairs"] = pairs

    pressure_rigid_rows = _pair_rows(pressure_rigid_pair_counts)
    required_pressure_rigid = [[cavity, "rigid_driver_displacement"]]
    if case in {"A", "B", "C", "D"}:
        required_pressure_rigid.append([cavity, "rigid_comparison_equalizer"])
    if case == "D":
        required_pressure_rigid.append(["air_port", "rigid_comparison_equalizer"])
    if case == "E":
        required_pressure_rigid.extend(
            [[cavity, "rigid_pr_back_mechanism"], ["air_rear_free", "rigid_pr_back_mechanism"]]
        )
    pressure_rigid_lookup = {tuple(row["domains"]): int(row["edge_count"]) for row in pressure_rigid_rows}
    missing_pressure_rigid = [
        pair for pair in required_pressure_rigid if tuple(sorted(pair)) not in pressure_rigid_lookup
    ]

    def line_vertices(name: str) -> set[int]:
        tag = next((tag for tag, group_name in line_names.items() if group_name == name), None)
        if tag is None:
            return set()
        return {int(node) for line, line_tag in zip(lines, line_tags) if int(line_tag) == tag for node in line}

    front_nodes = line_vertices(REFERENCE_PLANAR_PISTON_FRONT)
    back_nodes = line_vertices(REFERENCE_PLANAR_PISTON_BACK)
    piston_trace = {
        "front_group": REFERENCE_PLANAR_PISTON_FRONT,
        "back_group": REFERENCE_PLANAR_PISTON_BACK,
        "front_line_count": int(np.count_nonzero(line_tags == next((tag for tag, name in line_names.items() if name == REFERENCE_PLANAR_PISTON_FRONT), -1))),
        "back_line_count": int(np.count_nonzero(line_tags == next((tag for tag, name in line_names.items() if name == REFERENCE_PLANAR_PISTON_BACK), -1))),
        "front_unique_node_count": len(front_nodes),
        "back_unique_node_count": len(back_nodes),
        "shared_node_count": len(front_nodes & back_nodes),
        "separated": bool(front_nodes and back_nodes and not (front_nodes & back_nodes)),
        "identity": "reference planar piston",
    }
    pr_trace: dict[str, Any] = {"applicable": case == "E"}
    if case == "E":
        cavity_face_nodes = line_vertices(PR_CAVITY_FACE)
        exterior_face_nodes = line_vertices(PR_EXTERIOR_FACE)
        pr_trace.update(
            {
                "cavity_face_group": PR_CAVITY_FACE,
                "exterior_face_group": PR_EXTERIOR_FACE,
                "cavity_face_unique_node_count": len(cavity_face_nodes),
                "exterior_face_unique_node_count": len(exterior_face_nodes),
                "shared_node_count": len(cavity_face_nodes & exterior_face_nodes),
                "separated": bool(cavity_face_nodes and exterior_face_nodes and not (cavity_face_nodes & exterior_face_nodes)),
            }
        )
    else:
        pr_trace["separated"] = True

    negative_r_points = np.flatnonzero(points[:, 0] < -NEGATIVE_R_TOLERANCE_M)
    negative_r_triangles = np.flatnonzero(np.any(triangles_rz[:, :, 0] < -NEGATIVE_R_TOLERANCE_M, axis=1))
    degenerate_triangles = np.flatnonzero(areas <= AREA_TOLERANCE_M2)
    pressure_boundary_edges = {
        edge
        for edge, indices in edge_to_triangles.items()
        if len(indices) == 1
        and tag_to_name.get(int(triangle_tags[indices[0]]), "").startswith(PRESSURE_DOMAIN_PREFIX)
    }

    all_pressure_nonempty = all(
        domain_stats.get(name, {}).get("triangle_count", 0) > 0 for name in pressure_names
    )
    all_rigid_positive = all(
        domain_stats.get(name, {}).get("axisymmetric_volume_m3", 0.0) > 0.0 for name in rigid_names
    )
    line_groups_complete = (
        not missing_boundaries
        and not unknown_line_tags
        and all(row.get("line_count", 0) > 0 for row in line_group_stats.values())
    )
    named_groups_complete = (
        not missing_domains
        and not missing_boundaries
        and not unexpected_domains
        and not unexpected_boundaries
        and not unknown_triangle_tags
        and not unknown_line_tags
    )
    interfaces_clean = (
        not nonmanifold_edges
        and not uncovered_boundary_edges
        and not duplicate_line_edges
        and all(row["unmatched_line_count"] == 0 for row in line_group_stats.values())
        and all(row["duplicate_geometric_edge_count"] == 0 for row in line_group_stats.values())
    )
    volume_target = float(config["net_volume_target_m3"])
    cavity_volume = float(domain_stats.get(cavity, {}).get("axisymmetric_volume_m3", 0.0))
    volume_error = cavity_volume - volume_target
    relative_volume_error = abs(volume_error) / volume_target if volume_target > 0.0 else math.inf
    volume_contract = {
        "target_net_volume_m3": volume_target,
        "net_air_definition": "air_cavity only; rigid displacement domains and air_port are excluded",
        "air_cavity_volume_m3": cavity_volume,
        "air_cavity_error_m3": float(volume_error),
        "air_cavity_relative_error": float(relative_volume_error),
        "relative_tolerance": float(volume_relative_tolerance),
        "within_tolerance": bool(relative_volume_error <= volume_relative_tolerance),
        "air_port_volume_m3": float(domain_stats.get("air_port", {}).get("axisymmetric_volume_m3", 0.0)),
        "rigid_displacement_volumes_m3": {
            name: float(domain_stats.get(name, {}).get("axisymmetric_volume_m3", 0.0))
            for name in rigid_names
        },
        "pressure_domain_volume_total_m3": float(
            sum(domain_stats[name]["axisymmetric_volume_m3"] for name in pressure_names if name in domain_stats)
        ),
        "rigid_domain_volume_total_m3": float(
            sum(domain_stats[name]["axisymmetric_volume_m3"] for name in rigid_names if name in domain_stats)
        ),
    }

    checks = {
        "named_groups_complete": bool(named_groups_complete),
        "nonnegative_r": bool(len(negative_r_points) == 0 and len(negative_r_triangles) == 0),
        "positive_non_degenerate_triangles": bool(len(triangles) > 0 and len(degenerate_triangles) == 0),
        "minimum_triangle_quality": bool(len(qualities) > 0 and float(np.min(qualities)) >= minimum_triangle_quality),
        "pressure_domains_nonempty": bool(all_pressure_nonempty),
        "rigid_displacement_domains_positive_volume": bool(all_rigid_positive),
        "domain_components_single": bool(
            all(stats["component_count"] == 1 for stats in domain_stats.values() if stats["triangle_count"] > 0)
        ),
        "line_groups_complete": bool(line_groups_complete),
        "edge_adjacency_clean": bool(interfaces_clean),
        "pressure_rigid_interfaces_complete": bool(not missing_pressure_rigid),
        "reference_planar_piston_traces_separate": bool(piston_trace["separated"]),
        "passive_radiator_traces_separate": bool(pr_trace["separated"]),
        "pressure_connectivity_expected": bool(pressure_connectivity["required_case_connectivity"]),
        "net_cavity_volume_target": bool(volume_contract["within_tolerance"]),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)

    mesh_bytes = path.read_bytes()
    report: dict[str, Any] = {
        "schema": "luna.enclosure_mesh_topology_audit.v1",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "case_id": case,
        "source": {
            "mesh_path": str(path),
            "mesh_sha256": _sha256_bytes(mesh_bytes),
            "geometry_signature_sha256": _geometry_signature(mesh),
            "config": config,
        },
        "labels": {
            "pressure_domains": pressure_names,
            "rigid_displacement_domains": rigid_names,
            "reference_planar_piston": {
                "front": REFERENCE_PLANAR_PISTON_FRONT,
                "back": REFERENCE_PLANAR_PISTON_BACK,
                "identity": "reference planar piston",
            },
            "final_production_interface_ready": False,
        },
        "mesh_counts": {
            "points": int(len(points)),
            "line_elements": int(len(lines)),
            "triangle_elements": int(len(triangles)),
            "physical_1d_groups": int(len(line_names)),
            "physical_2d_groups": int(len(domain_names_by_tag)),
        },
        "quality": {
            "minimum_triangle_area_m2": float(np.min(areas)) if len(areas) else 0.0,
            "minimum_triangle_quality": float(np.min(qualities)) if len(qualities) else 0.0,
            "minimum_quality_threshold": float(minimum_triangle_quality),
            "area_tolerance_m2": AREA_TOLERANCE_M2,
            "degenerate_triangle_count": int(len(degenerate_triangles)),
            "negative_signed_area_triangle_count": int(np.count_nonzero(signed_cross < 0.0)),
            "negative_r_point_count": int(len(negative_r_points)),
            "negative_r_triangle_count": int(len(negative_r_triangles)),
            "minimum_r_m": float(np.min(points[:, 0])),
            "maximum_r_m": float(np.max(points[:, 0])),
        },
        "domains": {
            "pressure_domains": {name: domain_stats.get(name, {"triangle_count": 0}) for name in pressure_names},
            "rigid_displacement_domains": {name: domain_stats.get(name, {"triangle_count": 0}) for name in rigid_names},
        },
        "interfaces": {
            "geometric_edge_count": int(len(edge_to_triangles)),
            "triangle_edge_adjacency_histogram": _histogram(triangle_edge_adjacencies),
            "same_domain_interior_edge_count": int(same_domain_edge_count),
            "single_triangle_boundary_edge_count": int(len(single_triangle_edges)),
            "nonmanifold_edge_count": int(len(nonmanifold_edges)),
            "uncovered_pressure_boundary_edge_count": int(len(uncovered_boundary_edges)),
            "duplicate_line_edge_count": int(len(duplicate_line_edges)),
            "pressure_pressure_interfaces": _pair_rows(pressure_pair_counts),
            "pressure_rigid_interfaces": pressure_rigid_rows,
            "required_pressure_rigid_interfaces": [
                {"domains": sorted(pair), "edge_count": pressure_rigid_lookup.get(tuple(sorted(pair)), 0)}
                for pair in required_pressure_rigid
            ],
            "missing_pressure_rigid_interfaces": [sorted(pair) for pair in missing_pressure_rigid],
            "physical_boundary_groups": {
                name: line_group_stats[name] for name in sorted(line_group_stats)
            },
            "uncovered_boundary_edges": [list(edge) for edge in uncovered_boundary_edges],
            "nonmanifold_edges": [list(edge) for edge in nonmanifold_edges],
            "duplicate_line_edges": [list(edge) for edge in duplicate_line_edges],
        },
        "connectivity": pressure_connectivity,
        "traces": {
            "reference_planar_piston": piston_trace,
            "passive_radiator": pr_trace,
        },
        "volume_contract": volume_contract,
        "group_contract": {
            "expected_domain_names": sorted(expected_domains),
            "expected_boundary_names": sorted(expected_boundaries),
            "missing_domains": missing_domains,
            "missing_boundaries": missing_boundaries,
            "unexpected_domains": unexpected_domains,
            "unexpected_boundaries": unexpected_boundaries,
            "unknown_triangle_physical_tags": unknown_triangle_tags,
            "unknown_line_physical_tags": unknown_line_tags,
            "duplicate_field_data_tags": {key: sorted(set(value)) for key, value in sorted(duplicate_tags.items())},
        },
        "checks": checks,
    }
    return report


def audit_reference_mesh(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility spelling for callers that name the mesh explicitly."""

    return audit_mesh(*args, **kwargs)


def write_audit_json(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a stable, human-readable audit JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
