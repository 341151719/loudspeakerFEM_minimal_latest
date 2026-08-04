#!/usr/bin/env python3
"""Convert a COMSOL mesh-export MPHTXT file into a tagged Gmsh mesh.

COMSOL stores mapped quadrilaterals in tensor-product corner order
``[q00, q10, q01, q11]``.  The current solver is triangular, so each quad is
split along the q00-q11 diagonal while retaining the official mapped grid
lines and geometric entity IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import meshio
import numpy as np


def _data_lines(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            out.append(value)
    return out


def read_comsol_mesh(path: str | Path):
    lines = _data_lines(Path(path))
    mesh_marker = lines.index("4 Mesh")
    i = mesh_marker + 1
    i += 3  # mesh version, space dimension, number of vertices
    nvertices = int(lines[i - 1].split()[0])
    i += 1  # lowest vertex index
    points = np.asarray(
        [[float(x) for x in lines[i + j].split()] for j in range(nvertices)],
        float,
    )
    i += nvertices
    ntypes = int(lines[i].split()[0]); i += 1
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(ntypes):
        kind = lines[i].split(maxsplit=1)[1]; i += 1
        nper = int(lines[i].split()[0]); i += 1
        nelem = int(lines[i].split()[0]); i += 1
        conn = np.asarray(
            [[int(x) for x in lines[i + j].split()] for j in range(nelem)],
            int,
        ).reshape(nelem, nper)
        i += nelem
        nentity = int(lines[i].split()[0]); i += 1
        if nentity != nelem:
            raise ValueError(f"{kind}: {nelem} elements but {nentity} entity IDs")
        entities = np.asarray([int(lines[i + j].split()[0]) for j in range(nelem)], int)
        i += nelem
        blocks[kind] = (conn, entities)
    return points, blocks


def convert(
    source: str | Path,
    destination: str | Path,
    *,
    scale: float = 1e-3,
    quad_diagonal: str = "main",
):
    points2, blocks = read_comsol_mesh(source)
    points3 = np.column_stack([points2 * float(scale), np.zeros(len(points2))])
    cells = []
    physical = []
    if "edg" in blocks:
        edges, tags = blocks["edg"]
        cells.append(("line", edges))
        physical.append(tags + 1)
    triangles = []
    tri_tags = []
    if "quad" in blocks:
        q, tags = blocks["quad"]
        # Tensor order [q00,q10,q01,q11].
        if quad_diagonal == "main":
            qa = np.vstack([q[:, [0, 1, 3]], q[:, [0, 3, 2]]])
            ta = np.tile(tags, 2)
        elif quad_diagonal == "anti":
            qa = np.vstack([q[:, [0, 1, 2]], q[:, [1, 3, 2]]])
            ta = np.tile(tags, 2)
        elif quad_diagonal == "alternating":
            even = np.arange(len(q)) % 2 == 0
            parts = [
                q[even][:, [0, 1, 3]], q[even][:, [0, 3, 2]],
                q[~even][:, [0, 1, 2]], q[~even][:, [1, 3, 2]],
            ]
            qa = np.vstack(parts)
            ta = np.concatenate([tags[even], tags[even], tags[~even], tags[~even]])
        elif quad_diagonal == "center":
            start = len(points3)
            centers = points3[q].mean(axis=1)
            points3 = np.vstack([points3, centers])
            c = np.arange(start, start + len(q), dtype=int)
            qa = np.vstack([
                np.column_stack([q[:, 0], q[:, 1], c]),
                np.column_stack([q[:, 1], q[:, 3], c]),
                np.column_stack([q[:, 3], q[:, 2], c]),
                np.column_stack([q[:, 2], q[:, 0], c]),
            ])
            ta = np.tile(tags, 4)
        else:
            raise ValueError("quad_diagonal must be main, anti, alternating or center")
        triangles.append(qa)
        tri_tags.append(ta)
    if "tri" in blocks:
        tri, tags = blocks["tri"]
        triangles.append(tri)
        tri_tags.append(tags)
    if triangles:
        cells.append(("triangle", np.vstack(triangles)))
        physical.append(np.concatenate(tri_tags))
    mesh = meshio.Mesh(
        points3,
        cells,
        cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
    )
    meshio.write(destination, mesh, file_format="gmsh22", binary=False)
    return {
        "vertices": int(len(points2)),
        "lines": int(len(blocks.get("edg", ([], []))[0])),
        "triangles_after_quad_split": int(sum(len(x) for x in triangles)),
        "domain_ids": sorted(map(int, np.unique(np.concatenate(tri_tags)))) if tri_tags else [],
        "boundary_ids": sorted(map(int, np.unique(blocks["edg"][1] + 1))) if "edg" in blocks else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--scale", type=float, default=1e-3, help="coordinate scale; COMSOL export is mm")
    parser.add_argument("--quad-diagonal", choices=["main", "anti", "alternating", "center"], default="main")
    args = parser.parse_args()
    print(convert(args.source, args.destination, scale=args.scale, quad_diagonal=args.quad_diagonal))


if __name__ == "__main__":
    main()
