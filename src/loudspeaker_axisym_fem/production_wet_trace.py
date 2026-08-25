"""Read-only extraction of the production acoustic/structural wet traces.

This module deliberately operates on the existing production ``.msh`` only.  It
does not import a solver, alter the production mesh, or infer a production
drive law.  The common edges are discovered from triangle-domain adjacency;
the boundary ids in the result are therefore evidence from the input mesh,
not a hard-coded interface selection.

Two area measures are reported because the historical production acceptance
numbers are projected axisymmetric areas (``2*pi*r*abs(dr)``), while the true
surface area requested for a wet trace is ``2*pi*r*ds``.  Keeping both values
visible prevents the projected number from being mistaken for the curved
surface area used by a future velocity integral.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import meshio
import numpy as np


DEFAULT_PRODUCTION_MESH = Path(
    "inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh"
)
DEFAULT_MPHTXT = Path("inputs/comsol_reference/Untitled.mphtxt")
STRUCTURAL_DOMAINS = frozenset({3, 21, 25})
FRONT_ACOUSTIC_DOMAIN = 4
REAR_ACOUSTIC_DOMAIN = 2
REFERENCE_PLANAR_PISTON_RADIUS_M = 0.045


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of a file without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mesh_cells(mesh: meshio.Mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    blocks = [block for block in mesh.cells if block.type == cell_type]
    if not blocks:
        return np.empty((0, 2 if cell_type == "line" else 3), dtype=np.int64), np.empty(0, dtype=np.int64)
    cells = np.concatenate([np.asarray(block.data, dtype=np.int64) for block in blocks], axis=0)
    try:
        tags = np.asarray(mesh.cell_data_dict["gmsh:physical"][cell_type], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"production mesh has no gmsh:physical tags for {cell_type}") from exc
    if len(tags) != len(cells):
        raise ValueError(f"{cell_type} cell/tag length mismatch: {len(cells)} != {len(tags)}")
    return cells, tags


def _edge_domain_adjacency(
    triangles: np.ndarray, triangle_domains: np.ndarray
) -> dict[tuple[int, int], tuple[int, ...]]:
    adjacency: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle, domain in zip(triangles, triangle_domains):
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            key = (first, second) if first < second else (second, first)
            adjacency[key].append(int(domain))
    return {key: tuple(sorted(values)) for key, values in adjacency.items()}


def _physical_name_map(mesh: meshio.Mesh, dimension: int) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in mesh.field_data.items():
        array = np.asarray(value).reshape(-1)
        if len(array) >= 2 and int(array[1]) == int(dimension):
            result[int(array[0])] = str(name)
    return result


def _edge_components(edges: Iterable[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Return deterministic connected edge components."""

    remaining = set(edges)
    components: list[list[tuple[int, int]]] = []
    node_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for edge in remaining:
        node_edges[edge[0]].add(edge)
        node_edges[edge[1]].add(edge)
    while remaining:
        seed = min(remaining)
        queue = deque([seed[0], seed[1]])
        nodes: set[int] = set()
        while queue:
            node = queue.popleft()
            if node in nodes:
                continue
            nodes.add(node)
            for edge in sorted(node_edges[node]):
                if edge in remaining:
                    queue.extend(edge)
        component = sorted(edge for edge in remaining if edge[0] in nodes or edge[1] in nodes)
        for edge in component:
            remaining.remove(edge)
        components.append(component)
    return components


def _ordered_polyline(component: list[tuple[int, int]]) -> list[int]:
    """Order a path deterministically; branch leftovers are appended as paths."""

    if not component:
        return []
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in component:
        adjacency[edge[0]].append(edge)
        adjacency[edge[1]].append(edge)
    unused = set(component)
    odd_nodes = sorted(node for node, edges in adjacency.items() if len(edges) % 2)
    current = odd_nodes[0] if odd_nodes else min(adjacency)
    ordered = [current]
    while unused:
        choices = sorted(
            edge for edge in adjacency[current] if edge in unused
        )
        if not choices:
            # A branch or a disconnected remainder is still represented
            # deterministically by starting its smallest remaining endpoint.
            edge = min(unused)
            current = edge[0]
            ordered.append(current)
            choices = [edge]
        edge = choices[0]
        unused.remove(edge)
        current = edge[1] if edge[0] == current else edge[0]
        ordered.append(current)
    return ordered


def _polyline_data(
    edge_set: set[tuple[int, int]], points_rz: np.ndarray
) -> tuple[list[list[list[float]]], list[float]]:
    polylines: list[list[list[float]]] = []
    all_lengths: list[float] = []
    for component in _edge_components(edge_set):
        ordered = _ordered_polyline(component)
        polylines.append(
            [
                [float(points_rz[node, 0]), float(points_rz[node, 1])]
                for node in ordered
            ]
        )
        all_lengths.append(
            float(
                sum(
                    np.linalg.norm(points_rz[b] - points_rz[a])
                    for a, b in component
                )
            )
        )
    return polylines, all_lengths


def _boundary_record(
    boundary_id: int,
    edge_set: set[tuple[int, int]],
    pair_records: list[tuple[tuple[int, ...], tuple[int, ...]]],
    points_rz: np.ndarray,
    boundary_names: dict[int, str],
) -> dict[str, Any]:
    polylines, component_lengths = _polyline_data(edge_set, points_rz)
    length_m = 0.0
    surface_area_m2 = 0.0
    projected_area_m2 = 0.0
    for first, second in sorted(edge_set):
        p0 = points_rz[first]
        p1 = points_rz[second]
        segment_length = float(np.linalg.norm(p1 - p0))
        mean_r = float(0.5 * (p0[0] + p1[0]))
        length_m += segment_length
        surface_area_m2 += 2.0 * math.pi * mean_r * segment_length
        projected_area_m2 += 2.0 * math.pi * mean_r * abs(float(p1[0] - p0[0]))
    nodes = sorted({node for edge in edge_set for node in edge})
    values = points_rz[nodes]
    domain_pairs = sorted(
        {
            (int(structural_domain), int(acoustic_domain))
            for structural, acoustic in pair_records
            for structural_domain in structural
            for acoustic_domain in acoustic
        }
    )
    entity = {
        "boundary_id": int(boundary_id),
        "entity_id": int(boundary_id),
        "physical_name": boundary_names.get(int(boundary_id), f"boundary_{boundary_id}"),
        "domain_pairs": [
            {"structural_domain": int(pair[0]), "acoustic_domain": int(pair[1])}
            for pair in domain_pairs
        ],
        "polyline_rz": polylines[0] if len(polylines) == 1 else None,
        "polylines_rz": polylines,
        "component_lengths_m": component_lengths,
        "length_2d_m": float(length_m),
        "rotational_surface_area_2pi_r_ds_m2": float(surface_area_m2),
        "axisymmetric_projected_area_2pi_r_abs_dr_m2": float(projected_area_m2),
        "r_range_m": [float(np.min(values[:, 0])), float(np.max(values[:, 0]))],
        "z_range_m": [float(np.min(values[:, 1])), float(np.max(values[:, 1]))],
        "edge_count": int(len(edge_set)),
    }
    entity["stable_hash_sha256"] = _canonical_hash(entity)
    return entity


def extract_production_wet_traces(
    mesh_path: str | Path = DEFAULT_PRODUCTION_MESH,
    *,
    mphtxt_path: str | Path | None = DEFAULT_MPHTXT,
    reference_radius_m: float = REFERENCE_PLANAR_PISTON_RADIUS_M,
) -> dict[str, Any]:
    """Extract front/rear structural-acoustic common edges from a production mesh."""

    mesh_path = Path(mesh_path)
    mesh = meshio.read(mesh_path)
    if mesh.points.shape[1] < 2:
        raise ValueError("production mesh must contain r,z coordinates")
    points_rz = np.asarray(mesh.points[:, :2], dtype=float)
    lines, line_tags = _mesh_cells(mesh, "line")
    triangles, triangle_domains = _mesh_cells(mesh, "triangle")
    edge_adjacency = _edge_domain_adjacency(triangles, triangle_domains)
    boundary_names = _physical_name_map(mesh, 1)
    candidates: dict[int, set[tuple[int, int]]] = defaultdict(set)
    pair_records: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(list)
    all_line_counts: Counter[int] = Counter()

    for segment, tag_value in zip(lines, line_tags):
        boundary_id = int(tag_value)
        all_line_counts[boundary_id] += 1
        first, second = (int(segment[0]), int(segment[1]))
        edge = (first, second) if first < second else (second, first)
        adjacent_domains = edge_adjacency.get(edge, ())
        structural = tuple(sorted(set(adjacent_domains) & STRUCTURAL_DOMAINS))
        acoustic = tuple(
            sorted(
                set(adjacent_domains)
                & {FRONT_ACOUSTIC_DOMAIN, REAR_ACOUSTIC_DOMAIN}
            )
        )
        if structural and acoustic:
            candidates[boundary_id].add(edge)
            pair_records[boundary_id].append((structural, acoustic))

    entities: list[dict[str, Any]] = []
    for boundary_id in sorted(candidates):
        entities.append(
            _boundary_record(
                boundary_id,
                candidates[boundary_id],
                pair_records[boundary_id],
                points_rz,
                boundary_names,
            )
        )

    side_entities: dict[str, list[dict[str, Any]]] = {"front": [], "rear": []}
    for entity in entities:
        acoustic_domains = {
            int(pair["acoustic_domain"]) for pair in entity["domain_pairs"]
        }
        if acoustic_domains == {FRONT_ACOUSTIC_DOMAIN}:
            side_entities["front"].append(entity)
        elif acoustic_domains == {REAR_ACOUSTIC_DOMAIN}:
            side_entities["rear"].append(entity)
        else:
            raise ValueError(
                f"boundary {entity['boundary_id']} has mixed/unexpected acoustic domains: {acoustic_domains}"
            )

    def summarize_side(side: str) -> dict[str, Any]:
        rows = side_entities[side]
        acoustic_domain = FRONT_ACOUSTIC_DOMAIN if side == "front" else REAR_ACOUSTIC_DOMAIN
        return {
            "side": side,
            "acoustic_domain": acoustic_domain,
            "boundary_ids": [int(row["boundary_id"]) for row in rows],
            "entities": rows,
            "length_2d_m": float(sum(row["length_2d_m"] for row in rows)),
            "rotational_surface_area_2pi_r_ds_m2": float(
                sum(row["rotational_surface_area_2pi_r_ds_m2"] for row in rows)
            ),
            "axisymmetric_projected_area_2pi_r_abs_dr_m2": float(
                sum(row["axisymmetric_projected_area_2pi_r_abs_dr_m2"] for row in rows)
            ),
            "r_range_m": [
                float(min(row["r_range_m"][0] for row in rows)),
                float(max(row["r_range_m"][1] for row in rows)),
            ],
            "z_range_m": [
                float(min(row["z_range_m"][0] for row in rows)),
                float(max(row["z_range_m"][1] for row in rows)),
            ],
        }

    source = {
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "mphtxt_path": str(Path(mphtxt_path)) if mphtxt_path is not None else None,
        "mphtxt_sha256": (
            sha256_file(mphtxt_path)
            if mphtxt_path is not None and Path(mphtxt_path).exists()
            else None
        ),
        "coordinate_convention": "mesh.points[:,0:2] = (r,z) in metres",
    }
    reference_area = math.pi * float(reference_radius_m) ** 2
    front = summarize_side("front")
    rear = summarize_side("rear")
    comparison = {
        "reference_planar_piston": {
            "identity": "reference planar piston",
            "radius_m": float(reference_radius_m),
            "area_pi_r2_m2": float(reference_area),
        },
        "production_wet_surface": {
            "front_rotational_surface_area_m2": front["rotational_surface_area_2pi_r_ds_m2"],
            "rear_rotational_surface_area_m2": rear["rotational_surface_area_2pi_r_ds_m2"],
            "front_projected_area_m2": front["axisymmetric_projected_area_2pi_r_abs_dr_m2"],
            "rear_projected_area_m2": rear["axisymmetric_projected_area_2pi_r_abs_dr_m2"],
        },
        "mismatch": True,
        "mismatch_reason": (
            "reference planar piston is a radius-0.045 m disk; production front/rear "
            "traces are separate curved multi-boundary surfaces and have different areas"
        ),
        "final_production_interface_ready": False,
        "required_stage3_mapping": [
            "map the real moving structural partition on each selected boundary",
            "derive the local outward normal consistently from adjacent domains",
            "provide the real displacement/velocity field and keep front/rear pressure DOFs separate",
        ],
    }
    report = {
        "schema": "luna.production_wet_trace_audit.v1",
        "status": "pass" if {row["boundary_id"] for row in entities} else "fail",
        "labels": {
            "interface_name": "production wet trace candidate",
            "reference_name": "reference planar piston",
            "final_production_interface_ready": False,
        },
        "source": source,
        "selection": {
            "structural_domains": sorted(STRUCTURAL_DOMAINS),
            "front_acoustic_domain": FRONT_ACOUSTIC_DOMAIN,
            "rear_acoustic_domain": REAR_ACOUSTIC_DOMAIN,
            "algorithm": "line physical group whose triangle edge has one selected structural and one selected acoustic domain",
        },
        "mesh_counts": {
            "points": int(len(points_rz)),
            "line_elements": int(len(lines)),
            "triangle_elements": int(len(triangles)),
            "negative_r_point_count": int(np.count_nonzero(points_rz[:, 0] < -1.0e-12)),
        },
        "front": front,
        "rear": rear,
        "stable_trace_hash_sha256": _canonical_hash(
            {
                "source_mesh_sha256": source["mesh_sha256"],
                "front_entities": [row["stable_hash_sha256"] for row in front["entities"]],
                "rear_entities": [row["stable_hash_sha256"] for row in rear["entities"]],
            }
        ),
        "comparison": comparison,
        "line_group_counts": {
            str(tag): int(count) for tag, count in sorted(all_line_counts.items())
        },
    }
    return report


def write_json(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a deterministic audit JSON to a caller-selected path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
