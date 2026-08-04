from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
import math
import re

from .json_utils import to_jsonable, write_json


@dataclass(frozen=True)
class Vertex:
    id: int
    r_mm: float
    z_mm: float
    domain: int
    tol: Optional[float]


@dataclass(frozen=True)
class Edge:
    id: int
    v1: int
    v2: int
    s1: int
    s2: int
    up_domain: int
    down_domain: int
    curve: int
    tol: Optional[float]


@dataclass(frozen=True)
class BezierCurve:
    id: int
    rational: bool
    degree: int
    control_points: Tuple[Tuple[float, ...], ...]

    def euclidean_control_points(self) -> List[Tuple[float, float]]:
        pts = []
        for p in self.control_points:
            if self.rational:
                x, y, w = p
                pts.append((x / w, y / w))
            else:
                x, y = p[:2]
                pts.append((x, y))
        return pts

    def point(self, t: float) -> Tuple[float, float]:
        n = self.degree
        cps = self.control_points
        if n == 1:
            if self.rational:
                x0, y0, w0 = cps[0]
                x1, y1, w1 = cps[1]
                x = (1 - t) * x0 + t * x1
                y = (1 - t) * y0 + t * y1
                w = (1 - t) * w0 + t * w1
                return (x / w, y / w)
            (x0, y0), (x1, y1) = cps[0][:2], cps[1][:2]
            return ((1 - t) * x0 + t * x1, (1 - t) * y0 + t * y1)
        if n == 2:
            b0 = (1 - t) ** 2
            b1 = 2 * (1 - t) * t
            b2 = t ** 2
            if self.rational:
                x = b0 * cps[0][0] + b1 * cps[1][0] + b2 * cps[2][0]
                y = b0 * cps[0][1] + b1 * cps[1][1] + b2 * cps[2][1]
                w = b0 * cps[0][2] + b1 * cps[1][2] + b2 * cps[2][2]
                return (x / w, y / w)
            x = b0 * cps[0][0] + b1 * cps[1][0] + b2 * cps[2][0]
            y = b0 * cps[0][1] + b1 * cps[1][1] + b2 * cps[2][1]
            return (x, y)
        # generic de Casteljau fallback
        pts = [list(p) for p in cps]
        if self.rational:
            for _ in range(n):
                pts = [[(1-t)*a + t*b for a, b in zip(pts[i], pts[i+1])] for i in range(len(pts)-1)]
            x, y, w = pts[0]
            return (x / w, y / w)
        else:
            for _ in range(n):
                pts = [[(1-t)*a + t*b for a, b in zip(pts[i], pts[i+1])] for i in range(len(pts)-1)]
            return (pts[0][0], pts[0][1])

    def sampled(self, n: Optional[int] = None) -> List[Tuple[float, float]]:
        if n is None:
            n = 2 if self.degree <= 1 else 12
        return [self.point(i / (n - 1)) for i in range(n)]


@dataclass
class DomainLoop:
    domain_id: int
    edge_ids: List[int]
    vertex_ids: List[int]
    signed_curve_ids: List[int]
    area_mm2: float
    centroid_rz_mm: Tuple[float, float]


@dataclass
class ComsolGeometry:
    vertices: Dict[int, Vertex]
    edges: Dict[int, Edge]
    curves: Dict[int, BezierCurve]

    def domains(self) -> List[int]:
        ds = set()
        for e in self.edges.values():
            if e.up_domain > 0:
                ds.add(e.up_domain)
            if e.down_domain > 0:
                ds.add(e.down_domain)
        return sorted(ds)

    def boundary_edges(self) -> List[Edge]:
        return [e for e in self.edges.values() if e.up_domain == 0 or e.down_domain == 0]

    def edges_for_domain(self, domain_id: int) -> List[Edge]:
        return [e for e in self.edges.values() if e.up_domain == domain_id or e.down_domain == domain_id]

    def curve_points_for_edge(self, edge: Edge, samples_for_quadratic: int = 16) -> List[Tuple[float, float]]:
        curve = self.curves[edge.curve]
        pts = curve.sampled(2 if curve.degree <= 1 else samples_for_quadratic)
        v1 = self.vertices[edge.v1]
        v2 = self.vertices[edge.v2]
        p1 = (v1.r_mm, v1.z_mm)
        p2 = (v2.r_mm, v2.z_mm)
        d_start = _dist2(pts[0], p1) + _dist2(pts[-1], p2)
        d_rev = _dist2(pts[0], p2) + _dist2(pts[-1], p1)
        if d_rev < d_start:
            pts = list(reversed(pts))
        return pts

    def ordered_loop_for_domain(self, domain_id: int) -> DomainLoop:
        domain_edges = self.edges_for_domain(domain_id)
        if not domain_edges:
            raise ValueError(f"domain {domain_id} has no edges")
        by_v: Dict[int, List[Edge]] = defaultdict(list)
        for e in domain_edges:
            by_v[e.v1].append(e)
            by_v[e.v2].append(e)
        bad = {v: len(es) for v, es in by_v.items() if len(es) != 2}
        if bad:
            raise ValueError(f"domain {domain_id} does not form a single 2-regular boundary graph: {bad}")
        start = min(by_v)
        edge_ids: List[int] = []
        verts: List[int] = [start]
        current_v = start
        prev_eid = None
        for _ in range(len(domain_edges)):
            candidates = [e for e in by_v[current_v] if e.id != prev_eid]
            if not candidates:
                break
            # deterministic choice; only ambiguous at start, select lower edge id.
            if prev_eid is None and len(candidates) > 1:
                e = sorted(candidates, key=lambda x: x.id)[0]
            else:
                e = candidates[0]
            edge_ids.append(e.id)
            next_v = e.v2 if e.v1 == current_v else e.v1
            verts.append(next_v)
            prev_eid = e.id
            current_v = next_v
            if current_v == start:
                break
        if len(edge_ids) != len(domain_edges):
            # Try all possible starting edge choices; useful for complex loops.
            for first in sorted(by_v[start], key=lambda e: e.id):
                edge_ids = []
                verts = [start]
                current_v = start
                prev_eid = None
                forced = first.id
                for k in range(len(domain_edges)):
                    candidates = [e for e in by_v[current_v] if e.id != prev_eid]
                    if not candidates:
                        break
                    if k == 0:
                        e = first
                    else:
                        e = candidates[0]
                    edge_ids.append(e.id)
                    next_v = e.v2 if e.v1 == current_v else e.v1
                    verts.append(next_v)
                    prev_eid = e.id
                    current_v = next_v
                    if current_v == start:
                        break
                if len(edge_ids) == len(domain_edges):
                    break
        pts = [(self.vertices[v].r_mm, self.vertices[v].z_mm) for v in verts]
        area = signed_polygon_area(pts)
        centroid = polygon_centroid(pts)
        signed_curves = []
        for eid, va, vb in zip(edge_ids, verts[:-1], verts[1:]):
            e = self.edges[eid]
            sign = 1 if (e.v1 == va and e.v2 == vb) else -1
            signed_curves.append(sign * e.curve)
        return DomainLoop(domain_id, edge_ids, verts, signed_curves, area, centroid)

    def all_domain_loops(self) -> Dict[int, DomainLoop]:
        return {d: self.ordered_loop_for_domain(d) for d in self.domains()}

    def inventory(self) -> dict:
        loops = self.all_domain_loops()
        return {
            "vertices": len(self.vertices),
            "edges_boundaries": len(self.edges),
            "curves": len(self.curves),
            "curve_degrees": {str(k): sum(1 for c in self.curves.values() if c.degree == k) for k in sorted({c.degree for c in self.curves.values()})},
            "rational_curves": sum(1 for c in self.curves.values() if c.rational),
            "domain_ids": self.domains(),
            "boundary_edges": [e.id for e in self.boundary_edges()],
            "domain_loops": {
                str(d): {
                    "edge_count": len(loop.edge_ids),
                    "edge_ids": loop.edge_ids,
                    "signed_curve_ids": loop.signed_curve_ids,
                    "area_mm2_signed": loop.area_mm2,
                    "area_mm2_abs": abs(loop.area_mm2),
                    "centroid_r_mm": loop.centroid_rz_mm[0],
                    "centroid_z_mm": loop.centroid_rz_mm[1],
                }
                for d, loop in loops.items()
            },
        }

    def plot_edges(self, path: str | Path, *, show_domain_labels: bool = True, figsize=(7, 9)) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        for e in self.edges.values():
            pts = self.curve_points_for_edge(e)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=0.7, color="black")
            mx, my = xs[len(xs)//2], ys[len(ys)//2]
            ax.text(mx, my, str(e.id), fontsize=5, color="tab:red")
        if show_domain_labels:
            for d, loop in self.all_domain_loops().items():
                c = loop.centroid_rz_mm
                ax.text(c[0], c[1], str(d), fontsize=8, color="tab:blue", ha="center", va="center")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title("COMSOL final geometry from mphtxt: boundary ids (red), domain ids (blue)")
        ax.grid(True, linewidth=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    def write_inventory(self, path: str | Path) -> None:
        write_json(path, self.inventory(), indent=2)

    def export_geo_polyline(self, path: str | Path, *, scale: float = 1e-3, samples_for_quadratic: int = 16) -> None:
        """Export a debug .geo using sampled COMSOL curves.

        This is a geometry reconstruction/debug aid. It preserves COMSOL boundary and
        domain IDs as physical groups.  It does not yet implement COMSOL's mapped mesh
        controls or boundary-layer mesh settings.
        """
        path = Path(path)
        # The safest conformal export creates one curve per COMSOL boundary using a spline
        # through sampled points.  Points are reused by coordinate identity to keep shared
        # vertices conformal.
        point_id_by_xy: Dict[Tuple[float, float], int] = {}
        point_lines: List[str] = []
        curve_lines: List[str] = []
        next_pid = 1
        def point_id(x_mm: float, z_mm: float) -> int:
            nonlocal next_pid
            key = (round(x_mm * scale, 14), round(z_mm * scale, 14))
            if key not in point_id_by_xy:
                point_id_by_xy[key] = next_pid
                point_lines.append(f"Point({next_pid}) = {{{x_mm*scale:.17g}, {z_mm*scale:.17g}, 0, 1e-3}};")
                next_pid += 1
            return point_id_by_xy[key]
        for e in self.edges.values():
            pts = self.curve_points_for_edge(e, samples_for_quadratic=samples_for_quadratic)
            pids = [point_id(x, z) for x, z in pts]
            if len(pids) == 2:
                curve_lines.append(f"Line({e.id}) = {{{pids[0]}, {pids[1]}}};")
            else:
                curve_lines.append(f"Spline({e.id}) = {{{', '.join(map(str, pids))}}};")
        loop_lines: List[str] = []
        physical_lines: List[str] = []
        for d, loop in self.all_domain_loops().items():
            loop_id = 1000 + d
            surf_id = 2000 + d
            signed_edges = []
            for eid, va, vb in zip(loop.edge_ids, loop.vertex_ids[:-1], loop.vertex_ids[1:]):
                e = self.edges[eid]
                signed_edges.append(str(eid if (e.v1 == va and e.v2 == vb) else -eid))
            loop_lines.append(f"Curve Loop({loop_id}) = {{{', '.join(signed_edges)}}};")
            loop_lines.append(f"Plane Surface({surf_id}) = {{{loop_id}}};")
            physical_lines.append(f"Physical Surface(\"domain_{d}\", {d}) = {{{surf_id}}};")
        for e in self.edges.values():
            physical_lines.append(f"Physical Curve(\"boundary_{e.id}\", {e.id}) = {{{e.id}}};")
        text = "// Generated from COMSOL mphtxt final geometry. Units: meters.\nSetFactory(\"Built-in\");\n" + "\n".join(point_lines + curve_lines + loop_lines + physical_lines) + "\n"
        path.write_text(text, encoding="utf-8")


    def export_geo_comsol_mesh(
        self,
        path: str | Path,
        *,
        scale: float = 1e-3,
        samples_for_quadratic: int = 16,
        global_hmax_m: float = 0.008575,
        global_hmin_m: float = 0.00025,
        local_h_05_domains: Sequence[int] = (8, 10, 11, 12, 17, 18, 19, 22),
        local_h_2mm_domains: Sequence[int] = (9, 13, 14, 15, 16, 20),
        local_h_4mm_domains: Sequence[int] = (3, 21, 25),
        boundary_layer_curves: Sequence[int] = (12, 53, 95, 96, 97, 98),
        boundary_layer_size_m: float = 20e-6,
        boundary_layer_tangent_size_m: float = 0.00025,
        corner_refinement_curves: Sequence[int] = (),
        corner_refinement_size_m: float = 24e-6,
        mesh_size_extend_from_boundary: bool = True,
        boundary_layer_thickness_m: float = 0.0006,
        boundary_layer_ratio: float = 1.22,
        boundary_layer_target_domains: Sequence[int] = (6, 23),
        boundary_layer_quads: bool = False,
        pml_distribution_curves: Sequence[int] = (87, 88),
        mapped_distribution_curves: Sequence[int] = (22, 38, 41, 45),
    ) -> None:
        """Export a COMSOL-like meshing .geo.

        This preserves the Stage-1 final geometry while adding the most important
        COMSOL mesh controls visible in the exported .m file:

        * global acoustic max size lam0/5 (8.575 mm at fmax=8 kHz)
        * 0.5 mm local mesh on narrow gaps and voice coil domains
        * 2 mm / 4 mm local mesh on selected structural domains
        * boundary-layer field on pole-piece / top-plate near-coil boundaries
          12, 53, 95, 96, 97, 98
        * transfinite line distributions on the same curve IDs COMSOL marks as
          two-element and eight-element distributions.

        It is still a Gmsh reconstruction, not COMSOL's internal mapped mesher,
        but it moves the magnetic mesh away from the generic polyline debug mesh
        toward the COMSOL mesh intent.
        """
        path = Path(path)
        point_id_by_xy: Dict[Tuple[float, float], int] = {}
        point_lines: List[str] = []
        curve_lines: List[str] = []
        next_pid = 1
        def point_id(x_mm: float, z_mm: float) -> int:
            nonlocal next_pid
            key = (round(x_mm * scale, 14), round(z_mm * scale, 14))
            if key not in point_id_by_xy:
                point_id_by_xy[key] = next_pid
                point_lines.append(f"Point({next_pid}) = {{{x_mm*scale:.17g}, {z_mm*scale:.17g}, 0, {global_hmax_m:.17g}}};")
                next_pid += 1
            return point_id_by_xy[key]
        # curve id -> point ids for later MeshSize selection
        curve_point_ids: Dict[int, List[int]] = {}
        for e in self.edges.values():
            pts = self.curve_points_for_edge(e, samples_for_quadratic=samples_for_quadratic)
            pids = [point_id(x, z) for x, z in pts]
            curve_point_ids[e.id] = pids
            if len(pids) == 2:
                curve_lines.append(f"Line({e.id}) = {{{pids[0]}, {pids[1]}}};")
            else:
                curve_lines.append(f"Spline({e.id}) = {{{', '.join(map(str, pids))}}};")
        loop_lines: List[str] = []
        physical_lines: List[str] = []
        for d, loop in self.all_domain_loops().items():
            loop_id = 1000 + d
            surf_id = 2000 + d
            signed_edges = []
            for eid, va, vb in zip(loop.edge_ids, loop.vertex_ids[:-1], loop.vertex_ids[1:]):
                e = self.edges[eid]
                signed_edges.append(str(eid if (e.v1 == va and e.v2 == vb) else -eid))
            loop_lines.append(f"Curve Loop({loop_id}) = {{{', '.join(signed_edges)}}};")
            loop_lines.append(f"Plane Surface({surf_id}) = {{{loop_id}}};")
            physical_lines.append(f"Physical Surface(\"domain_{d}\", {d}) = {{{surf_id}}};")
        for e in self.edges.values():
            physical_lines.append(f"Physical Curve(\"boundary_{e.id}\", {e.id}) = {{{e.id}}};")

        def points_for_domains(domains: Sequence[int]) -> List[int]:
            pts: set[int] = set()
            ds = set(int(x) for x in domains)
            for d in ds:
                try:
                    loop = self.ordered_loop_for_domain(d)
                except Exception:
                    continue
                for eid in loop.edge_ids:
                    pts.update(curve_point_ids.get(eid, []))
            return sorted(pts)

        mesh_lines: List[str] = [
            "// COMSOL-like mesh controls reconstructed from .m mesh feature settings",
            f"Mesh.CharacteristicLengthMin = {global_hmin_m:.17g};",
            f"Mesh.CharacteristicLengthMax = {global_hmax_m:.17g};",
            f"Mesh.MeshSizeExtendFromBoundary = {1 if mesh_size_extend_from_boundary else 0};",
            "Mesh.MeshSizeFromCurvature = 16;",
            "Mesh.Algorithm = 6; // Frontal-Delaunay",
        ]
        for domains, h, label in [
            (local_h_4mm_domains, 0.004, 'COMSOL size2 domains 3/21/25, hmax=4 mm'),
            (local_h_2mm_domains, 0.002, 'COMSOL size1 domains 9/13/14/15/16/20, hmax=2 mm'),
            (local_h_05_domains, 0.0005, 'COMSOL size3/narrow/coil domains, hmax=0.5 mm'),
        ]:
            pids = points_for_domains(domains)
            if pids:
                mesh_lines.append(f"// {label}")
                mesh_lines.append(f"MeshSize{{{', '.join(map(str, pids))}}} = {h:.17g};")
        bl_curves = [int(c) for c in boundary_layer_curves if int(c) in self.edges]
        if bl_curves:
            # Tangential resolution is independent of the much smaller first
            # layer normal size. Coupling both would create an unnecessarily
            # huge nearly isotropic mesh along the complete iron boundary.
            bl_pts = sorted({pid for c in bl_curves for pid in curve_point_ids.get(c, [])})
            if bl_pts:
                mesh_lines.append("// Boundary-layer tangential curve size")
                mesh_lines.append(f"MeshSize{{{', '.join(map(str, bl_pts))}}} = {boundary_layer_tangent_size_m:.17g};")
            corner_curves = [
                int(c) for c in corner_refinement_curves if int(c) in self.edges
            ]
            corner_pts = sorted({
                pid for c in corner_curves for pid in curve_point_ids.get(c, [])
            })
            if corner_pts:
                mesh_lines.append(
                    "// Isotropic refinement on curved iron corners; no offset layer"
                )
                mesh_lines.append(
                    f"MeshSize{{{', '.join(map(str, corner_pts))}}} = "
                    f"{corner_refinement_size_m:.17g};"
                )
            target_surfaces = {
                2000 + int(d) for d in boundary_layer_target_domains
                if int(d) in self.all_domain_loops()
            }
            excluded_surfaces = sorted(
                2000 + int(d) for d in self.all_domain_loops()
                if 2000 + int(d) not in target_surfaces
            )
            mesh_lines.extend([
                "// Gmsh BoundaryLayer field inside conducting soft-iron domains only",
                "Field[1] = BoundaryLayer;",
                f"Field[1].CurvesList = {{{', '.join(map(str, bl_curves))}}};",
                f"Field[1].Size = {boundary_layer_size_m:.17g};",
                f"Field[1].SizeFar = {max(0.002, boundary_layer_size_m*4):.17g};",
                f"Field[1].Thickness = {boundary_layer_thickness_m:.17g};",
                f"Field[1].Ratio = {boundary_layer_ratio:.17g};",
                f"Field[1].Quads = {1 if boundary_layer_quads else 0};",
                "Field[1].IntersectMetrics = 1;",
                *(
                    [f"Field[1].ExcludedSurfacesList = {{{', '.join(map(str, excluded_surfaces))}}};"]
                    if excluded_surfaces else []
                ),
                "BoundaryLayer Field = 1;",
            ])
        for curves, n, label in [
            (mapped_distribution_curves, 3, 'COMSOL Distribution numelem=2'),
            (pml_distribution_curves, 9, 'COMSOL PML Distribution numelem=8'),
        ]:
            kept = [int(c) for c in curves if int(c) in self.edges]
            if kept:
                mesh_lines.append(f"// {label}: transfinite curves use points = elements+1")
                mesh_lines.append(f"Transfinite Curve {{{', '.join(map(str, kept))}}} = {int(n)} Using Progression 1;")
        text = "// Generated from COMSOL mphtxt final geometry. Units: meters.\nSetFactory(\"Built-in\");\n" + "\n".join(point_lines + curve_lines + loop_lines + physical_lines + mesh_lines) + "\n"
        path.write_text(text, encoding="utf-8")


def _parse_float(token: str) -> Optional[float]:
    token = token.strip()
    if token.upper() == "NAN":
        return None
    return float(token)


def _dist2(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def signed_polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def polygon_centroid(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    area = signed_polygon_area(points)
    if abs(area) < 1e-12:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        c = x0 * y1 - x1 * y0
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    return (cx / (6.0 * area), cy / (6.0 * area))


def parse_mphtxt(path: str | Path) -> ComsolGeometry:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    vertices: Dict[int, Vertex] = {}
    edges: Dict[int, Edge] = {}
    curves: Dict[int, BezierCurve] = {}
    i = 0
    n_vertices = None
    while i < len(lines):
        line = lines[i]
        if "# number of vertices" in line:
            n_vertices = int(line.split()[0])
        if line.strip() == "# Vertices":
            # skip column header
            i += 2
            for vid in range(1, n_vertices + 1):
                parts = lines[i].split()
                vertices[vid] = Vertex(vid, float(parts[0]), float(parts[1]), int(parts[2]), _parse_float(parts[3]))
                i += 1
            continue
        if "# number of edges" in line:
            n_edges = int(line.split()[0])
        if line.strip() == "# Edges":
            i += 2
            for eid in range(1, n_edges + 1):
                parts = lines[i].split()
                edges[eid] = Edge(eid, int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), _parse_float(parts[7]))
                i += 1
            continue
        if "# number of curves" in line:
            n_curves = int(line.split()[0])
        if line.startswith("# Curve "):
            cid = int(line.split()[2])
            class_line = lines[i + 1]
            version_line = lines[i + 2]
            sdim = int(lines[i + 3].split()[0])
            rational = bool(int(lines[i + 4].split()[0]))
            degree = int(lines[i + 5].split()[0])
            cp_start = i + 7
            cp_count = degree + 1
            cps = []
            for k in range(cp_count):
                parts = lines[cp_start + k].split()
                cps.append(tuple(float(x) for x in parts[:3 if rational else 2]))
            curves[cid] = BezierCurve(cid, rational, degree, tuple(cps))
            i = cp_start + cp_count
            continue
        i += 1
    if not vertices or not edges or not curves:
        raise ValueError(f"failed to parse geometry from {path}")
    return ComsolGeometry(vertices, edges, curves)


DOMAIN_SELECTIONS_FROM_M = {
    "PML": [1, 5],
    "Soft Iron": [6, 23],
    "Composite": [3, 21],
    "Cloth": [20],
    "Foam": [25],
    "Coil": [17, 18, 19],
    "Glass Fiber": [9, 10, 11, 12, 13, 14, 15, 16],
    "Generic Ferrite": [24],
    "Air": [2, 4, 7, 8, 22],
    "Structural Domains": [3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25],
    "Composite and Glass Fiber": [3, 9, 10, 11, 12, 13, 14, 15, 16, 21],
    "Magnetic Domains": [6, 17, 18, 19, 23, 24],
}

BOUNDARY_SELECTIONS_FROM_M = {
    "Exterior field / radiated power Boundary 93": [93],
    "Fixed spider/surround boundaries": [81, 85],
    "Mapped mesh distribution 2 elems": [22, 38, 41, 45],
    "PML distribution 8 elems": [87, 88],
    "Boundary layer iron/gap boundaries": [12, 53, 95, 96, 97, 98],
    "Boundary layer exterior field boundary": [93],
}
