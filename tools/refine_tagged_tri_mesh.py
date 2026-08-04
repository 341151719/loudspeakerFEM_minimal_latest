#!/usr/bin/env python3
"""Uniformly refine a tagged Gmsh triangle mesh without losing entity IDs.

Every triangle is split into four children and every tagged boundary line into
two children.  Midpoints are shared globally, so nonconforming cracks are not
introduced.  This is used for native structural mesh-convergence tests; it is
not a COMSOL-data correction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import meshio
import numpy as np


def _blocks(mesh: meshio.Mesh, kind: str) -> tuple[np.ndarray, np.ndarray]:
    cells: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    pdata = mesh.cell_data.get("gmsh:physical", [])
    for i, block in enumerate(mesh.cells):
        if block.type != kind:
            continue
        cells.append(np.asarray(block.data, dtype=int))
        if i >= len(pdata):
            raise ValueError(f"cell block {i} ({kind}) has no gmsh:physical data")
        tags.append(np.asarray(pdata[i], dtype=int))
    if not cells:
        return np.empty((0, 2 if kind == "line" else 3), int), np.empty(0, int)
    return np.vstack(cells), np.concatenate(tags)


def refine(source: str | Path, destination: str | Path, levels: int = 1, domains: set[int] | None = None) -> dict:
    mesh = meshio.read(source)
    points = np.asarray(mesh.points, float)
    lines, line_tags = _blocks(mesh, "line")
    triangles, tri_tags = _blocks(mesh, "triangle")
    if not len(triangles):
        raise ValueError("source mesh has no triangle cells")

    for _ in range(int(levels)):
        pts = points.tolist()
        midpoint_ids: dict[tuple[int, int], int] = {}
        marked_edges: set[tuple[int, int]] | None = None
        if domains is not None:
            marked_edges = set()
            for (a, b, c), tag in zip(triangles, tri_tags):
                if int(tag) in domains:
                    marked_edges.update((tuple(sorted((int(a), int(b)))),
                                         tuple(sorted((int(b), int(c)))),
                                         tuple(sorted((int(c), int(a))))))

        def midpoint(a: int, b: int) -> int:
            edge = tuple(sorted((int(a), int(b))))
            idx = midpoint_ids.get(edge)
            if idx is None:
                idx = len(pts)
                midpoint_ids[edge] = idx
                pts.append((0.5 * (points[edge[0]] + points[edge[1]])).tolist())
            return idx

        children: list[list[int]] = []
        child_tags: list[int] = []
        for (a, b, c), tag in zip(triangles, tri_tags):
            edges = [tuple(sorted((int(a), int(b)))), tuple(sorted((int(b), int(c)))), tuple(sorted((int(c), int(a))))]
            flags = [marked_edges is None or e in marked_edges for e in edges]
            nsplit = sum(flags)
            ab = midpoint(a, b) if flags[0] else None
            bc = midpoint(b, c) if flags[1] else None
            ca = midpoint(c, a) if flags[2] else None
            if nsplit == 0:
                local = [[a, b, c]]
            elif nsplit == 1:
                if flags[0]: local = [[a, ab, c], [ab, b, c]]
                elif flags[1]: local = [[b, bc, a], [bc, c, a]]
                else: local = [[c, ca, b], [ca, a, b]]
            elif nsplit == 2:
                if flags[0] and flags[1]: local = [[b, bc, ab], [a, ab, c], [ab, bc, c]]
                elif flags[1] and flags[2]: local = [[c, ca, bc], [b, bc, a], [bc, ca, a]]
                else: local = [[a, ab, ca], [b, c, ab], [ab, c, ca]]
            else:
                local = [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            children.extend(local)
            child_tags.extend([int(tag)] * len(local))
        split_lines: list[list[int]] = []
        split_line_tags: list[int] = []
        for (a, b), tag in zip(lines, line_tags):
            edge = tuple(sorted((int(a), int(b))))
            if marked_edges is None or edge in marked_edges:
                ab = midpoint(a, b)
                split_lines.extend([[a, ab], [ab, b]])
                split_line_tags.extend([int(tag), int(tag)])
            else:
                split_lines.append([a, b]); split_line_tags.append(int(tag))

        points = np.asarray(pts, float)
        triangles = np.asarray(children, int)
        tri_tags = np.asarray(child_tags, int)
        lines = np.asarray(split_lines, int)
        line_tags = np.asarray(split_line_tags, int)

    cells = [("line", lines), ("triangle", triangles)]
    physical = [line_tags, tri_tags]
    out = meshio.Mesh(
        points,
        cells,
        cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
        field_data=mesh.field_data,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(destination, out, file_format="gmsh22", binary=False)
    return {
        "source": str(source),
        "destination": str(destination),
        "levels": int(levels),
        "selective_domains": None if domains is None else sorted(map(int, domains)),
        "points": int(len(points)),
        "lines": int(len(lines)),
        "triangles": int(len(triangles)),
        "boundary_ids": sorted(map(int, np.unique(line_tags))),
        "domain_ids": sorted(map(int, np.unique(tri_tags))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--domains", help="comma-separated domain IDs for conforming red/green local refinement")
    args = parser.parse_args()
    domains = None if not args.domains else {int(x) for x in args.domains.split(",")}
    print(refine(args.source, args.destination, args.levels, domains))


if __name__ == "__main__":
    main()
