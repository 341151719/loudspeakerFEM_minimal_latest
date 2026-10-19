from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .core import (
    AIR_DOMAINS, SOLID_DOMAINS, SHELL_TAGS, SHELL_GROUPS, SYMMETRY,
    rows, atomic_npz, guard
)

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (0, 2), (1, 2))
FACES = {
    "tet": ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
    "pyr": ((0, 1, 3, 2), (0, 1, 4), (1, 3, 4),
            (3, 2, 4), (2, 0, 4)),
    "hex": ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
}


@dataclass
class Mesh:
    x: np.ndarray
    tet: np.ndarray
    domain: np.ndarray
    parent: np.ndarray
    btri: np.ndarray
    tag: np.ndarray
    adjacency: dict[int, list[int]]
    sym_edges: np.ndarray
    metadata: dict
    t10: np.ndarray | None = None
    b6: np.ndarray | None = None
    sym3: np.ndarray | None = None


@dataclass
class Space:
    snode: np.ndarray
    scell: np.ndarray
    pnode: np.ndarray
    pcell: np.ndarray
    hnode: np.ndarray
    hcell: np.ndarray

    @property
    def offset(self) -> int:
        return 3 * len(self.snode)

    @property
    def nm(self) -> int:
        return self.offset + 6 * len(self.hnode)

    @property
    def np(self) -> int:
        return len(self.pnode)

    def sdofs(self, ci: int) -> np.ndarray:
        return (3 * self.scell[ci, :, None] + np.arange(3)).ravel()

    def hdofs(self, bi: int) -> np.ndarray:
        return (self.offset + 6 * self.hcell[bi, :, None]
                + np.arange(6)).ravel()


def integer_list(text: str) -> list[int]:
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)) or any(type(v) is not int for v in value):
        raise ValueError(f"Expected a literal integer list, received {text!r}")
    return list(value)


def _conn(root: Path, kind: str) -> list[tuple]:
    result = []
    for i, r in enumerate(rows(root / f"elements_{kind}.csv")):
        keys = sorted((k for k in r if k.startswith("n")),
                      key=lambda k: int(k[1:]))
        result.append((kind, int(r["entity"]), [int(r[k]) for k in keys], i))
    return result


def native_faces(cells: list[tuple]) -> dict:
    registry = {}
    for ci, (kind, domain, nodes, _) in enumerate(cells):
        for local in FACES[kind]:
            face = tuple(nodes[i] for i in local)
            key = tuple(sorted(face))
            item = registry.setdefault(key, [face, []])
            item[1].append((ci, domain))
            if len(item[1]) > 2:
                raise ValueError(f"Nonmanifold native volume face {key}")
    return registry


def _valid_polyhedron(x: np.ndarray, kind: str, nodes: list[int],
                      label: str) -> None:
    p = x[nodes]
    h = max(np.linalg.norm(a - b) for a, b in combinations(p, 2))
    if h <= 0:
        raise ValueError(f"{label}: coincident cell corners")
    center = p.mean(axis=0)
    # Exported COMSOL corners contain sub-nanometre roundoff on nominally
    # planar quads (maximum observed warp 1.9e-7 of the cell diameter).
    tol = 1e-6 * h
    for face in FACES[kind]:
        q = p[list(face)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        area2 = np.linalg.norm(n)
        if area2 <= 1e-14 * h * h:
            raise ValueError(f"{label}: degenerate face {face}")
        n /= area2
        if np.dot(n, center - q[0]) > 0:
            n = -n
        if np.max(np.abs((q - q[0]) @ n)) > tol:
            raise ValueError(f"{label}: warped face; straight conversion refused")
        if np.max((p - q[0]) @ n) > tol:
            raise ValueError(f"{label}: nonconvex cell or incorrect tensor ordering")
        if np.dot(center - q[0], n) >= -1e-14 * h:
            raise ValueError(f"{label}: cell center is not strictly interior")


def orient_tets(x: np.ndarray, tet: np.ndarray) -> tuple[np.ndarray, int]:
    t = np.asarray(tet, dtype=np.int64).copy()
    p = x[t]
    det = np.linalg.det(np.stack((p[:, 1] - p[:, 0],
                                 p[:, 2] - p[:, 0],
                                 p[:, 3] - p[:, 0]), axis=2))
    h = np.max(np.linalg.norm(p[:, :, None] - p[:, None, :], axis=3),
               axis=(1, 2))
    bad = np.where(np.abs(det) <= 1e-14 * h**3)[0]
    if len(bad):
        raise ValueError(f"Geometrically degenerate tetrahedra {bad[:12].tolist()}")
    negative = det < 0
    old = t[negative, 0].copy()
    t[negative, 0] = t[negative, 1]
    t[negative, 1] = old
    return t, int(negative.sum())


def convert(x: np.ndarray, cells: list[tuple], surfaces: list[tuple],
            adjacency: dict[int, list[int]], sym_edges: np.ndarray,
            metadata: dict | None = None) -> Mesh:
    """Input cell corners use the native tensor ordering, not VTK ordering."""
    x = np.asarray(x, float)
    registry = native_faces(cells)
    points = x.tolist()
    face_center = {}
    for key, (face, _) in registry.items():
        if len(face) == 4:
            face_center[key] = len(points)
            points.append(x[list(face)].mean(axis=0).tolist())

    def triangles(face):
        if len(face) == 3:
            return [list(face)]
        c = face_center[tuple(sorted(face))]
        return [[face[j], face[(j + 1) % 4], c] for j in range(4)]

    tets, domains, parents = [], [], []
    type_id = {"tet": 0, "pyr": 1, "hex": 2}
    for ci, (kind, domain, nodes, original_row) in enumerate(cells):
        if min(nodes) < 0 or max(nodes) >= len(x):
            raise ValueError(f"{kind} row {original_row}: node index out of range")
        _valid_polyhedron(x, kind, nodes, f"{kind} row {original_row}, domain {domain}")
        if kind == "tet":
            children = [nodes]
        else:
            center = len(points)
            points.append(x[nodes].mean(axis=0).tolist())
            children = []
            for lf in FACES[kind]:
                face = tuple(nodes[j] for j in lf)
                children.extend([[center, *q] for q in triangles(face)])
        tets.extend(children)
        domains.extend([domain] * len(children))
        parents.extend([[type_id[kind], original_row]] * len(children))

    btri, tags, seen = [], [], set()
    for kind, tag, nodes, row in surfaces:
        face = tuple(nodes if kind == "tri" else [nodes[j] for j in (0, 1, 3, 2)])
        key = tuple(sorted(face))
        if key in seen:
            raise ValueError(f"Duplicate tagged surface facet: boundary {tag}, row {row}")
        seen.add(key)
        if key not in registry:
            raise ValueError(f"Boundary {tag}, row {row}: no native volume face")
        domains_here = [d for _, d in registry[key][1]]
        expected = set(domains_here + ([0] if len(domains_here) == 1 else []))
        if set(adjacency[tag]) != expected:
            raise ValueError(
                f"Boundary {tag}: geometric adjacency {adjacency[tag]} "
                f"does not match mesh adjacency {sorted(expected)}"
            )
        for q in triangles(face):
            btri.append(q)
            tags.append(tag)
    missing = [key for key, (_, adj) in registry.items()
               if len(adj) == 1 and key not in seen]
    if missing:
        raise ValueError(f"Untagged exterior native faces: {missing[:8]}")
    xx = np.asarray(points)
    tt, flips = orient_tets(xx, np.asarray(tets))
    meta = dict(metadata or {})
    meta["orientation_permutations"] = flips
    meta["conversion"] = "shared quad center and cell center; tet corners retained"
    m = Mesh(xx, tt, np.array(domains), np.array(parents),
             np.asarray(btri, dtype=np.int64), np.array(tags), adjacency,
             np.asarray(sym_edges, dtype=np.int64).reshape(-1, 2), meta)
    audit_incidence(m)
    return m


def load(root: Path, cfg: dict) -> Mesh:
    vr = rows(root / "vertices.csv")
    ids = np.array([int(r["node"]) for r in vr])
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("vertices.csv must contain contiguous, ordered zero-based nodes")
    x = 1e-3 * np.array([[float(r[k]) for k in ("x_mm", "y_mm", "z_mm")]
                         for r in vr])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite mesh coordinates")
    raw = {k: _conn(root, k) for k in ("tet", "pyr", "hex", "tri", "quad", "edg", "vtx")}
    adjacency = {int(r["boundary"]): integer_list(r["adjacent_domains_including_void"])
                 for r in rows(root / "boundary_adjacency.csv")}
    selections = {}
    for r in rows(root / "selections.csv"):
        selections[r["tag"]] = (int(r["dimension"]), integer_list(r["entities"]))
    if "box2" not in selections or selections["box2"][0] != 1:
        raise ValueError("box2 must exist and must be an edge selection")
    if "box1" not in selections or selections["box1"][0] != 2:
        raise ValueError("box1 must exist and must be a boundary selection")
    if set(selections["box1"][1]) != SYMMETRY:
        raise ValueError("box1 differs from the documented symmetry selection")
    expected_selections = {
        "dif1": AIR_DOMAINS, "uni1": SOLID_DOMAINS, "uni3": SHELL_TAGS,
        **{f"sel{i + 9}": v for i, v in enumerate(SHELL_GROUPS.values())}
    }
    for tag, entities in expected_selections.items():
        if tag not in selections or set(selections[tag][1]) != entities:
            raise ValueError(f"Selection {tag} differs from the benchmark assignment")
    sym_edges = [r[2] for r in raw["edg"] if r[1] in selections["box2"][1]]
    counts = {k: len(v) for k, v in raw.items()}
    counts["vertices"] = len(x)
    if cfg["strict_bundle"]:
        expected = dict(tet=56332, pyr=2369, hex=2422, tri=6572,
                        quad=4194, edg=2643, vtx=290, vertices=14176)
        if counts != expected:
            raise ValueError(f"Native count mismatch: {counts}; expected {expected}")
        if set(adjacency) != set(range(1, 278)):
            raise ValueError("Expected boundary entities 1 through 277")
        domains = {r[1] for k in ("tet", "pyr", "hex") for r in raw[k]}
        if domains != set(range(1, 43)):
            raise ValueError("Expected volume domains 1 through 42")
        if x[:, 1].max() > 1e-10 or abs(x[:, 1].max()) > 1e-8:
            raise ValueError("Expected the negative-Y half geometry")
    # Metadata expressions are never evaluated.
    metadata = {
        "raw_counts": counts, "selections": selections,
        "materials_metadata": rows(root / "materials.csv"),
        "parameters_metadata": rows(root / "parameters.csv"),
        "live_settings": rows(root / "live_settings.csv"),
        "coordinate_input_unit": "mm", "coordinate_internal_unit": "m",
        "native_shell_foldline_literal": "0.001; unit not inferred"
    }
    m = convert(x, raw["tet"] + raw["pyr"] + raw["hex"],
                raw["tri"] + raw["quad"], adjacency,
                np.asarray(sym_edges), metadata)
    for _ in range(int(cfg["level"])):
        guard(cfg, len(m.tet) * 8 * 1300 + len(m.btri) * 4 * 1000,
              "refinement working storage estimate")
        m = refine(m)
    return elevate(m)


def incidence(m: Mesh) -> dict:
    result = {}
    for ci, t in enumerate(m.tet):
        center = m.x[t].mean(axis=0)
        for lf in FACES["tet"]:
            face = [int(t[j]) for j in lf]
            p = m.x[face]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]),
                      center - p[0]) > 0:
                face[1], face[2] = face[2], face[1]
            result.setdefault(tuple(sorted(face)), []).append((ci, face))
    return result


def audit_incidence(m: Mesh) -> dict:
    inc = incidence(m)
    tagged = {tuple(sorted(q)): int(tag) for q, tag in zip(m.btri, m.tag)}
    if len(tagged) != len(m.btri):
        raise ValueError("Duplicate converted boundary triangles")
    for key, adj in inc.items():
        if len(adj) not in (1, 2):
            raise ValueError(f"Converted nonmanifold face {key}")
        if len(adj) == 1 and key not in tagged:
            raise ValueError(f"Untagged converted exterior face {key}")
    for key, tag in tagged.items():
        if key not in inc:
            raise ValueError(f"Converted boundary {tag} has no volume face")
        a = inc[key]
        found = {int(m.domain[ci]) for ci, _ in a}
        if len(a) == 1:
            found.add(0)
        if found != set(m.adjacency[tag]):
            raise ValueError(f"Converted adjacency mismatch on boundary {tag}")
    return inc


def refine(m: Mesh) -> Mesh:
    if m.t10 is not None:
        raise ValueError("Refine corner geometry before quadratic elevation")
    x = m.x.tolist()
    mids = {}

    def mid(a, b):
        key = tuple(sorted((int(a), int(b))))
        if key not in mids:
            mids[key] = len(x)
            x.append(((m.x[a] + m.x[b]) / 2).tolist())
        return mids[key]

    tt, dd, pp = [], [], []
    for t, d, p in zip(m.tet, m.domain, m.parent):
        a, b, c, e = map(int, t)
        ab, ac, ae, bc, be, ce = [mid(t[i], t[j]) for i, j in EDGES]
        children = [
            [a, ab, ac, ae], [b, ab, bc, be], [c, ac, bc, ce],
            [e, ae, be, ce], [ab, ac, ae, ce], [ab, ac, bc, ce],
            [ab, ae, be, ce], [ab, bc, be, ce]
        ]
        tt.extend(children)
        dd.extend([d] * 8)
        pp.extend([p] * 8)
    bb, tags = [], []
    for q, tag in zip(m.btri, m.tag):
        a, b, c = map(int, q)
        ab, ac, bc = mid(a, b), mid(a, c), mid(b, c)
        bb.extend([[a, ab, ac], [b, bc, ab], [c, ac, bc], [ab, bc, ac]])
        tags.extend([tag] * 4)
    ee = [[a, mid(a, b)] for a, b in m.sym_edges]
    ee += [[mid(a, b), b] for a, b in m.sym_edges]
    xx = np.asarray(x)
    t, _ = orient_tets(xx, np.asarray(tt))
    meta = dict(m.metadata)
    meta["uniform_refinements"] = meta.get("uniform_refinements", 0) + 1
    result = Mesh(xx, t, np.array(dd), np.array(pp), np.array(bb),
                  np.array(tags), m.adjacency, np.asarray(ee).reshape(-1, 2), meta)
    audit_incidence(result)
    return result


def elevate(m: Mesh) -> Mesh:
    x = m.x.tolist()
    mids = {}

    def mid(a, b):
        key = tuple(sorted((int(a), int(b))))
        if key not in mids:
            mids[key] = len(x)
            x.append(((m.x[a] + m.x[b]) / 2).tolist())
        return mids[key]

    t10 = np.array([[*t, *[mid(t[i], t[j]) for i, j in EDGES]] for t in m.tet])
    b6 = np.array([[*t, *[mid(t[i], t[j]) for i, j in TRI_EDGES]] for t in m.btri])
    sym3 = np.array([[a, b, mid(a, b)] for a, b in m.sym_edges],
                    dtype=np.int64).reshape(-1, 3)
    return Mesh(np.asarray(x), m.tet, m.domain, m.parent, m.btri, m.tag,
                m.adjacency, m.sym_edges, m.metadata, t10, b6, sym3)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = int(self.p[i])
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def number_spaces(m: Mesh, inc: dict | None = None) -> Space:
    inc = incidence(m) if inc is None else inc
    barriers = {tuple(sorted(q)) for q, tag in zip(m.btri, m.tag)
                if tag in SHELL_TAGS}

    def continuous(domains, barrier):
        active = np.flatnonzero(np.isin(m.domain, list(domains)))
        slot = np.full(len(m.tet), -1, int)
        slot[active] = np.arange(len(active))
        uf = UnionFind(10 * len(active))
        for key, adj in inc.items():
            if len(adj) != 2 or (barrier and key in barriers):
                continue
            a, b = adj[0][0], adj[1][0]
            if slot[a] < 0 or slot[b] < 0:
                continue
            common = set(m.t10[a]).intersection(m.t10[b])
            for g in common:
                ia = int(np.flatnonzero(m.t10[a] == g)[0])
                ib = int(np.flatnonzero(m.t10[b] == g)[0])
                uf.join(10 * slot[a] + ia, 10 * slot[b] + ib)
        roots, nodes = {}, []
        cell = np.full((len(m.tet), 10), -1, int)
        for ci in active:
            for j, g in enumerate(m.t10[ci]):
                root = uf.find(10 * slot[ci] + j)
                if root not in roots:
                    roots[root] = len(nodes)
                    nodes.append(g)
                cell[ci, j] = roots[root]
        return np.array(nodes, int), cell

    sn, sc = continuous(SOLID_DOMAINS, False)
    pn, pc = continuous(AIR_DOMAINS, True)
    mask = np.isin(m.tag, list(SHELL_TAGS))
    hn = np.unique(m.b6[mask])
    lookup = np.full(len(m.x), -1, int)
    lookup[hn] = np.arange(len(hn))
    hc = np.full(m.b6.shape, -1, int)
    hc[mask] = lookup[m.b6[mask]]
    return Space(sn, sc, pn, pc, hn, hc)


def trace_positions(m: Mesh, ci: int, face6: np.ndarray) -> np.ndarray:
    return np.array([int(np.flatnonzero(m.t10[ci] == g)[0]) for g in face6])


class Locator:
    """Exact barycentric containment; KD tree only prunes candidates."""
    def __init__(self, m: Mesh, cells: np.ndarray):
        self.m = m
        self.cells = np.asarray(cells, int)
        p = m.x[m.tet[self.cells]]
        self.origin = p[:, 0]
        self.inv = np.linalg.inv(np.stack((p[:, 1] - p[:, 0],
                                          p[:, 2] - p[:, 0],
                                          p[:, 3] - p[:, 0]), axis=2))
        self.center = p.mean(axis=1)
        self.radius = np.linalg.norm(p - self.center[:, None], axis=2).max(axis=1)
        self.max_radius = float(self.radius.max(initial=0))
        self.tree = cKDTree(self.center)
        self.domain_trees = {}
        for domain in np.unique(m.domain[self.cells]):
            indices = np.flatnonzero(m.domain[self.cells] == domain)
            self.domain_trees[int(domain)] = (
                indices, cKDTree(self.center[indices]),
                float(self.radius[indices].max(initial=0)))

    def bary(self, local: int, p: np.ndarray) -> np.ndarray:
        z = self.inv[local] @ (p - self.origin[local])
        return np.r_[1 - z.sum(), z]

    def locate(self, p: np.ndarray, tolerance: float = 1e-9) -> tuple[int, np.ndarray, float]:
        candidates = self.tree.query_ball_point(p, self.max_radius + tolerance)
        best = None
        for j in candidates:
            if np.linalg.norm(p - self.center[j]) > self.radius[j] + tolerance:
                continue
            L = self.bary(j, p)
            error = max(0.0, -float(L.min())) * max(self.radius[j], 1e-30)
            if error <= tolerance and (best is None or error < best[2]):
                best = (int(self.cells[j]), L, error)
        if best is None:
            raise ValueError(f"No containing element for point {p.tolist()} "
                             f"within {tolerance:g} m")
        return best

    def thickness_segments(self, p: np.ndarray, n: np.ndarray, h: float,
                           domains: set[int]) -> list[tuple[float, float, int]]:
        intervals = []
        candidates = []
        for domain in domains:
            entry = self.domain_trees.get(int(domain))
            if entry is None:
                continue
            indices, tree, max_radius = entry
            candidates.extend(indices[tree.query_ball_point(p, max_radius + h / 2)])
        for j in candidates:
            if np.linalg.norm(p - self.center[j]) > self.radius[j] + h / 2:
                continue
            ci = int(self.cells[j])
            L = self.bary(j, p)
            g = self.inv[j] @ n
            dL = np.r_[-g.sum(), g]
            lo, hi = -h / 2, h / 2
            valid = True
            for a, b in zip(L, dL):
                if abs(b) < 1e-14 / max(h, self.radius[j]):
                    if a < -2e-11:
                        valid = False
                        break
                elif b > 0:
                    lo = max(lo, (-2e-11 - a) / b)
                else:
                    hi = min(hi, (-2e-11 - a) / b)
            if valid and hi - lo > 1e-11 * h:
                intervals.append((lo, hi, ci))
        if not intervals:
            return []
        cuts = sorted(set(v for a, b, _ in intervals for v in (a, b)))
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            z = (a + b) / 2
            owners = [ci for lo, hi, ci in intervals if lo <= z <= hi]
            if owners and b - a > 1e-11 * h:
                pieces.append((a, b, min(owners)))
        # Retain only the connected material interval containing the midsurface.
        groups = []
        for piece in pieces:
            if not groups or piece[0] > groups[-1][-1][1] + 1e-9 * h:
                groups.append([piece])
            else:
                groups[-1].append(piece)
        for group in groups:
            if group[0][0] <= 1e-9 * h and group[-1][1] >= -1e-9 * h:
                return group
        return []


def save_geometry(path: Path, m: Mesh, s: Space) -> None:
    atomic_npz(
        path, x=m.x, tet=m.tet, domain=m.domain, parent=m.parent,
        btri=m.btri, tag=m.tag, sym_edges=m.sym_edges,
        t10=m.t10, b6=m.b6, sym3=m.sym3,
        snode=s.snode, scell=s.scell, pnode=s.pnode, pcell=s.pcell,
        hnode=s.hnode, hcell=s.hcell,
        metadata=np.array(json.dumps({
            "adjacency": m.adjacency, "metadata": m.metadata
        }))
    )


def restore_geometry(path: Path) -> tuple[Mesh, Space]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["metadata"]))
    adj = {int(k): v for k, v in meta["adjacency"].items()}
    m = Mesh(d["x"], d["tet"], d["domain"], d["parent"], d["btri"],
             d["tag"], adj, d["sym_edges"], meta["metadata"],
             d["t10"], d["b6"], d["sym3"])
    s = Space(d["snode"], d["scell"], d["pnode"], d["pcell"],
              d["hnode"], d["hcell"])
    return m, s
