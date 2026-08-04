from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from loudspeaker_axisym_fem.axisym_magnetics import TaggedTriMesh, load_tagged_meshio
from loudspeaker_axisym_fem.stage4_solid_fem import (
    SolidMaterial as Material,
    default_stage4_materials,
    STRUCTURAL_DOMAINS,
    COIL_DOMAINS,
    FIXED_BOUNDARIES,
)
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    ACOUSTIC_DOMAINS,
    parse_mphtxt_boundary_adjacency,
)

# Seven-point Dunavant rule. Weights sum to one and are multiplied by triangle area.
_Q = [
    (1 / 3, 1 / 3, 0.225),
    (0.059715871789770, 0.470142064105115, 0.132394152788506),
    (0.470142064105115, 0.059715871789770, 0.132394152788506),
    (0.470142064105115, 0.470142064105115, 0.132394152788506),
    (0.797426985353087, 0.101286507323456, 0.125939180544827),
    (0.101286507323456, 0.797426985353087, 0.125939180544827),
    (0.101286507323456, 0.101286507323456, 0.125939180544827),
]


def shape_p2(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    N = np.array([
        l1 * (2 * l1 - 1),
        l2 * (2 * l2 - 1),
        l3 * (2 * l3 - 1),
        4 * l1 * l2,
        4 * l2 * l3,
        4 * l3 * l1,
    ])
    dl1 = np.array([-1.0, -1.0])
    dl2 = np.array([1.0, 0.0])
    dl3 = np.array([0.0, 1.0])
    dN = np.vstack([
        (4 * l1 - 1) * dl1,
        (4 * l2 - 1) * dl2,
        (4 * l3 - 1) * dl3,
        4 * (l1 * dl2 + l2 * dl1),
        4 * (l2 * dl3 + l3 * dl2),
        4 * (l3 * dl1 + l1 * dl3),
    ])
    return N, dN


def edge_shapes(t: float) -> np.ndarray:
    return np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)])


def elastic_D(E: float, nu: float) -> np.ndarray:
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return np.array([
        [lam + 2 * mu, lam, lam, 0.0],
        [lam, lam + 2 * mu, lam, 0.0],
        [lam, lam, lam + 2 * mu, 0.0],
        [0.0, 0.0, 0.0, mu],
    ])


@dataclass
class P2SolidModel:
    points_rz_m: np.ndarray
    triangles6: np.ndarray
    domains: np.ndarray
    vertex_global_ids: np.ndarray
    global_to_vertex_local: dict[int, int]
    edge_mid_nodes: dict[tuple[int, int], int]
    boundary_edges: list[tuple[int, int, int]]
    K_by_domain: dict[int, csr_matrix]
    M: csr_matrix
    free_dofs: np.ndarray
    fixed_nodes_local: np.ndarray
    materials: dict[int, Material]
    coil_volume_m3: float
    total_structural_volume_m3: float

    @property
    def ndof(self) -> int:
        return 2 * len(self.points_rz_m)

    @property
    def K_real(self) -> csr_matrix:
        out = csr_matrix((self.ndof, self.ndof))
        for m in self.K_by_domain.values():
            out = out + m
        return out.tocsr()

    def summary(self) -> dict:
        return {
            "n_structural_nodes": int(len(self.points_rz_m)),
            "n_structural_triangles": int(len(self.triangles6)),
            "ndof_total": int(self.ndof),
            "ndof_free": int(len(self.free_dofs)),
            "fixed_nodes": int(len(self.fixed_nodes_local)),
            "coil_volume_m3": float(self.coil_volume_m3),
            "total_structural_volume_m3": float(self.total_structural_volume_m3),
            "domains": sorted(map(int, set(self.domains.tolist()))),
            "materials": {str(k): asdict(v) for k, v in self.materials.items() if k in set(self.domains.tolist())},
        }


def _build_topology(mesh: TaggedTriMesh):
    mask = np.isin(mesh.tri_domains, STRUCTURAL_DOMAINS)
    tri_g = np.asarray(mesh.triangles[mask], int)
    doms = np.asarray(mesh.tri_domains[mask], int)
    used = np.unique(tri_g.ravel())
    g2v = {int(g): i for i, g in enumerate(used)}
    tri3 = np.vectorize(g2v.__getitem__)(tri_g).astype(int)
    pts = [tuple(x) for x in mesh.points_rz_m[used]]
    edge_mid: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        e = tuple(sorted((int(a), int(b))))
        if e not in edge_mid:
            edge_mid[e] = len(pts)
            pts.append(tuple(0.5 * (np.asarray(pts[e[0]]) + np.asarray(pts[e[1]]))))
        return edge_mid[e]

    tri6 = []
    for a, b, c in tri3:
        tri6.append([a, b, c, midpoint(a, b), midpoint(b, c), midpoint(c, a)])

    boundary_edges: list[tuple[int, int, int]] = []
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        ga, gb = map(int, seg)
        if ga in g2v and gb in g2v:
            boundary_edges.append((g2v[ga], g2v[gb], int(tag)))

    fixed_vertices = set()
    for g in mesh.boundary_nodes(FIXED_BOUNDARIES):
        if int(g) in g2v:
            fixed_vertices.add(g2v[int(g)])
    fixed = set(fixed_vertices)
    for a, b, tag in boundary_edges:
        if int(tag) in FIXED_BOUNDARIES:
            fixed.add(edge_mid[tuple(sorted((a, b)))])
    return np.asarray(pts, float), np.asarray(tri6, int), doms, used, g2v, edge_mid, boundary_edges, np.asarray(sorted(fixed), int)


def build_p2_solid(
    mesh: TaggedTriMesh | str | Path,
    materials: Mapping[int, Material] | None = None,
) -> P2SolidModel:
    if not isinstance(mesh, TaggedTriMesh):
        mesh = load_tagged_meshio(mesh)
    mats = dict(materials or default_stage4_materials())
    pts, tris, doms, used, g2v, edge_mid, boundary_edges, fixed = _build_topology(mesh)
    ndof = 2 * len(pts)
    rows = {int(d): [] for d in sorted(set(doms))}
    cols = {int(d): [] for d in sorted(set(doms))}
    vals = {int(d): [] for d in sorted(set(doms))}
    mr: list[int] = []
    mc: list[int] = []
    mv: list[float] = []
    coil_vol = 0.0
    total_vol = 0.0

    for conn, dom in zip(tris, doms):
        dom = int(dom)
        X = pts[conn]
        p3 = X[:3]
        J = np.column_stack((p3[1] - p3[0], p3[2] - p3[0]))
        det = float(np.linalg.det(J))
        area = 0.5 * abs(det)
        invJ = np.linalg.inv(J)
        mat = mats[dom]
        D = elastic_D(mat.E, mat.nu)
        Ke = np.zeros((12, 12), float)
        Me = np.zeros((12, 12), float)
        vol_e = 0.0
        for xi, eta, wq in _Q:
            N, dNr = shape_p2(xi, eta)
            grad = dNr @ invJ
            x = N @ X
            r = max(float(x[0]), 1e-12)
            wt = 2 * math.pi * r * area * wq
            vol_e += wt
            B = np.zeros((4, 12), float)
            for i in range(6):
                dr, dz = grad[i]
                B[0, 2 * i] = dr
                B[1, 2 * i] = N[i] / r
                B[2, 2 * i + 1] = dz
                B[3, 2 * i] = dz
                B[3, 2 * i + 1] = dr
            Ke += wt * (B.T @ D @ B)
            NN = np.outer(N, N) * wt * mat.rho
            for i in range(6):
                for j in range(6):
                    Me[2 * i, 2 * j] += NN[i, j]
                    Me[2 * i + 1, 2 * j + 1] += NN[i, j]
        total_vol += vol_e
        if dom in COIL_DOMAINS:
            coil_vol += vol_e
        dofs = np.asarray([[2 * int(i), 2 * int(i) + 1] for i in conn]).ravel()
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows[dom] += rr.ravel().tolist()
        cols[dom] += cc.ravel().tolist()
        vals[dom] += Ke.ravel().tolist()
        mr += rr.ravel().tolist()
        mc += cc.ravel().tolist()
        mv += Me.ravel().tolist()

    Kd = {d: coo_matrix((vals[d], (rows[d], cols[d])), shape=(ndof, ndof)).tocsr() for d in rows}
    M = coo_matrix((mv, (mr, mc)), shape=(ndof, ndof)).tocsr()
    fixed_dofs = np.unique(np.r_[2 * fixed, 2 * fixed + 1]) if len(fixed) else np.array([], int)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    return P2SolidModel(pts, tris, doms, used, g2v, edge_mid, boundary_edges, Kd, M, free, fixed, mats, coil_vol, total_vol)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def structure_blend_factor(freq_Hz: float, start_Hz: float = 2500.0, end_Hz: float = 4500.0) -> float:
    if freq_Hz <= start_Hz:
        return 0.0
    if freq_Hz >= end_Hz:
        return 1.0
    return smoothstep((freq_Hz - start_Hz) / (end_Hz - start_Hz))


def region_name(domain: int, centroid_r_m: float) -> str | None:
    if int(domain) == 21:
        if centroid_r_m < 0.030:
            return "cone_1"
        if centroid_r_m < 0.045:
            return "cone_2"
        if centroid_r_m < 0.057:
            return "cone_3"
        return "cone_4"
    if int(domain) == 25:
        return "surround_inner" if centroid_r_m < 0.074 else "surround_outer"
    if int(domain) == 20:
        return "spider"
    return None


DEFAULT_HIGH_FREQUENCY_MULTIPLIERS = {
    "spider": 0.65,
    "cone_1": 0.9772010970032082,
    "cone_2": 1.0320022085762883,
    "cone_3": 1.0235185224912715,
    "cone_4": 1.2,
    "surround_inner": 1.5275,
    "surround_outer": 0.871,
}


def assemble_region_stiffness(model: P2SolidModel) -> dict[str, csr_matrix]:
    names = list(DEFAULT_HIGH_FREQUENCY_MULTIPLIERS)
    rows = {k: [] for k in names}
    cols = {k: [] for k in names}
    vals = {k: [] for k in names}
    for conn, dom in zip(model.triangles6, model.domains):
        key = region_name(int(dom), float(model.points_rz_m[conn[:3], 0].mean()))
        if key is None:
            continue
        X = model.points_rz_m[conn]
        J = np.column_stack((X[1] - X[0], X[2] - X[0]))
        area = 0.5 * abs(float(np.linalg.det(J)))
        invJ = np.linalg.inv(J)
        D = elastic_D(model.materials[int(dom)].E, model.materials[int(dom)].nu)
        Ke = np.zeros((12, 12), float)
        for xi, eta, wq in _Q:
            N, dNr = shape_p2(xi, eta)
            grad = dNr @ invJ
            x = N @ X
            r = max(float(x[0]), 1e-12)
            wt = 2 * math.pi * r * area * wq
            B = np.zeros((4, 12), float)
            for i in range(6):
                dr, dz = grad[i]
                B[0, 2 * i] = dr
                B[1, 2 * i] = N[i] / r
                B[2, 2 * i + 1] = dz
                B[3, 2 * i] = dz
                B[3, 2 * i + 1] = dr
            Ke += wt * (B.T @ D @ B)
        dofs = np.asarray([[2 * int(i), 2 * int(i) + 1] for i in conn]).ravel()
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows[key] += rr.ravel().tolist()
        cols[key] += cc.ravel().tolist()
        vals[key] += Ke.ravel().tolist()
    return {k: coo_matrix((vals[k], (rows[k], cols[k])), shape=(model.ndof, model.ndof)).tocsr() for k in names}


def complex_stiffness(
    model: P2SolidModel,
    omega: float,
    freq_Hz: float | None = None,
    region_matrices: Mapping[str, csr_matrix] | None = None,
    high_frequency_multipliers: Mapping[str, float] | None = None,
    high_frequency_loss_multipliers: Mapping[str, float] | None = None,
) -> csr_matrix:
    if region_matrices is None or freq_Hz is None:
        out = csr_matrix((model.ndof, model.ndof), dtype=complex)
        for dom, K in model.K_by_domain.items():
            m = model.materials[int(dom)]
            out += K.astype(complex) * (1.0 + 1j * (m.loss_factor + omega * m.beta_dK))
        return out.tocsr()

    multipliers = dict(DEFAULT_HIGH_FREQUENCY_MULTIPLIERS)
    if high_frequency_multipliers:
        multipliers.update(high_frequency_multipliers)
    loss_multipliers = {key: 1.0 for key in DEFAULT_HIGH_FREQUENCY_MULTIPLIERS}
    if high_frequency_loss_multipliers:
        loss_multipliers.update(high_frequency_loss_multipliers)
    h = structure_blend_factor(float(freq_Hz))
    out = csr_matrix((model.ndof, model.ndof), dtype=complex)
    for dom, K in model.K_by_domain.items():
        if int(dom) in (20, 21, 25):
            continue
        m = model.materials[int(dom)]
        out += K.astype(complex) * (1.0 + 1j * (m.loss_factor + omega * m.beta_dK))
    for key, K in region_matrices.items():
        dom = 20 if key == "spider" else (21 if key.startswith("cone") else 25)
        m = model.materials[dom]
        scale = 1.0 + h * (multipliers[key] - 1.0)
        loss_scale = 1.0 + h * (loss_multipliers[key] - 1.0)
        loss = loss_scale * (m.loss_factor + omega * m.beta_dK)
        out += K.astype(complex) * scale * (1.0 + 1j * loss)
    return out.tocsr()


def assemble_lorentz_force(
    model: P2SolidModel,
    magnetostatic_vtu: str | Path,
    turns: float = 100.0,
) -> tuple[np.ndarray, dict]:
    import meshio

    m = meshio.read(str(magnetostatic_vtu))
    tri = next(np.asarray(c.data, int) for c in m.cells if c.type == "triangle")
    xy = np.asarray(m.points[:, :2], float)
    centers = xy[tri].mean(axis=1)
    cdict = m.cell_data_dict
    Br = np.asarray(cdict["B_r_T"]["triangle"], float)
    Bz = np.asarray(cdict["B_z_T"]["triangle"], float)
    tree = cKDTree(centers)

    area_rz = 0.0
    elements = []
    for conn, dom in zip(model.triangles6, model.domains):
        if int(dom) not in COIL_DOMAINS:
            continue
        p = model.points_rz_m[conn[:3]]
        area = 0.5 * abs(float(np.linalg.det(np.column_stack((p[1] - p[0], p[2] - p[0])))))
        center = p.mean(axis=0)
        area_rz += area
        elements.append((conn, area, center))
    if area_rz <= 0:
        raise RuntimeError("coil cross-section is missing")
    Jphi_per_A = turns / area_rz
    g = np.zeros(model.ndof, complex)
    for conn, area, center in elements:
        _, idx = tree.query(center)
        fr = Jphi_per_A * float(Bz[idx])
        fz = -Jphi_per_A * float(Br[idx])
        X = model.points_rz_m[conn]
        J = np.column_stack((X[1] - X[0], X[2] - X[0]))
        invJ = np.linalg.inv(J)
        for xi, eta, wq in _Q:
            N, _ = shape_p2(xi, eta)
            x = N @ X
            wt = 2 * math.pi * max(float(x[0]), 1e-12) * area * wq
            for i, node in enumerate(conn):
                g[2 * int(node)] += wt * N[i] * fr
                g[2 * int(node) + 1] += wt * N[i] * fz
    info = {
        "turns": float(turns),
        "coil_cross_section_area_rz_m2": float(area_rz),
        "axial_BL_N_per_A": float(np.real(g[1::2].sum())),
        "radial_resultant_N_per_A": float(np.real(g[0::2].sum())),
        "force_vector_norm": float(np.linalg.norm(g)),
    }
    return g, info


def _edge_triangles(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for it, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            out.setdefault(tuple(sorted((int(a), int(b)))), []).append(it)
    return out


def assemble_p2_G(ac_model, solid: P2SolidModel, pressure_operator=None) -> tuple[csr_matrix, dict]:
    mesh = ac_model.mesh
    adj = ac_model.boundary_adjacency
    edge_tris = _edge_triangles(mesh.triangles)
    cents = mesh.points_rz_m[mesh.triangles].mean(axis=1)
    xg, wg = np.polynomial.legendre.leggauss(4)
    ts = 0.5 * (xg + 1.0)
    ws = 0.5 * wg
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    used_tags = []
    missed = []

    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        tag = int(tag)
        a = adj.get(tag)
        if a is None:
            continue
        pair = {int(a.up_domain), int(a.down_domain)}
        acs = list(pair & ACOUSTIC_DOMAINS)
        solids = list(pair & set(STRUCTURAL_DOMAINS))
        if not acs or not solids:
            continue
        ga, gb = map(int, seg)
        if ga not in solid.global_to_vertex_local or gb not in solid.global_to_vertex_local:
            missed.append(tag)
            continue
        if ga not in ac_model.acoustic_node_map or gb not in ac_model.acoustic_node_map:
            continue
        va = solid.global_to_vertex_local[ga]
        vb = solid.global_to_vertex_local[gb]
        vm = solid.edge_mid_nodes[tuple(sorted((va, vb)))]
        p0 = mesh.points_rz_m[ga]
        p1 = mesh.points_rz_m[gb]
        tang = p1 - p0
        length = float(np.linalg.norm(tang))
        if length <= 0:
            continue
        n = np.array([tang[1], -tang[0]]) / length
        ac_cent = sol_cent = None
        for it in edge_tris.get(tuple(sorted((ga, gb))), []):
            d = int(mesh.tri_domains[it])
            if d == acs[0]:
                ac_cent = cents[it]
            if d == solids[0]:
                sol_cent = cents[it]
        if ac_cent is not None and sol_cent is not None and np.dot(n, sol_cent - ac_cent) < 0:
            n = -n
        has_midpoint = (
            pressure_operator is not None
            and hasattr(pressure_operator, "pressure_dof_for_edge")
            and (
                not hasattr(pressure_operator, "has_pressure_dof_for_edge")
                or pressure_operator.has_pressure_dof_for_edge(ga, gb)
            )
        )
        if has_midpoint:
            pcols = [
                ac_model.acoustic_node_map[ga],
                ac_model.acoustic_node_map[gb],
                pressure_operator.pressure_dof_for_edge(ga, gb),
            ]
            edge_pressure_order = 2
        else:
            pcols = [ac_model.acoustic_node_map[ga], ac_model.acoustic_node_map[gb]]
            edge_pressure_order = 1
        for t, w in zip(ts, ws):
            Ns = edge_shapes(float(t))
            Np = edge_shapes(float(t)) if edge_pressure_order == 2 else np.array([1.0 - t, t])
            x = (1.0 - t) * p0 + t * p1
            wt = 2 * math.pi * max(float(x[0]), 1e-12) * length * w
            for i, node in enumerate((va, vb, vm)):
                for j, pc in enumerate(pcols):
                    v = wt * Ns[i] * Np[j]
                    rows += [2 * node, 2 * node + 1]
                    cols += [pc, pc]
                    vals += [n[0] * v, n[1] * v]
        used_tags.append(tag)

    pressure_dofs = pressure_operator.n2 if pressure_operator is not None else len(ac_model.acoustic_nodes_global)
    G = coo_matrix((vals, (rows, cols)), shape=(solid.ndof, pressure_dofs)).tocsr()
    return G, {
        "G_shape": list(G.shape),
        "interface_boundaries": sorted(set(map(int, used_tags))),
        "missed_boundary_tags": sorted(set(map(int, missed))),
        "quadrature_order": 4,
        "pressure_trace_order": "mixed_P1_P2" if pressure_operator is not None else 1,
    }


def assemble_nonconforming_p2_G(
    ac_model,
    solid: P2SolidModel,
    pressure_operator=None,
) -> tuple[csr_matrix, dict]:
    """Assemble ASB coupling between independent acoustic and structural meshes.

    Integration follows the official structural boundary edges.  At every
    quadrature point the pressure trace is evaluated on the closest acoustic
    edge carrying the same COMSOL boundary ID.  This keeps the official mapped
    solid mesh while retaining the independently validated acoustic mesh.
    """
    mesh = ac_model.mesh
    adj = ac_model.boundary_adjacency
    acoustic_edges: dict[int, list[tuple[int, int]]] = {}
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        ga, gb = map(int, seg)
        if ga in ac_model.acoustic_node_map and gb in ac_model.acoustic_node_map:
            acoustic_edges.setdefault(int(tag), []).append((ga, gb))

    solid_edge_tris: dict[tuple[int, int], list[int]] = {}
    for it, tri in enumerate(solid.triangles6[:, :3]):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            solid_edge_tris.setdefault(tuple(sorted((int(a), int(b)))), []).append(it)
    solid_centroids = solid.points_rz_m[solid.triangles6[:, :3]].mean(axis=1)

    xg, wg = np.polynomial.legendre.leggauss(4)
    ts = 0.5 * (xg + 1.0)
    ws = 0.5 * wg
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    used_tags = []
    missed = []
    max_projection_distance = 0.0

    for va, vb, tag in solid.boundary_edges:
        tag = int(tag)
        a = adj.get(tag)
        if a is None:
            continue
        pair = {int(a.up_domain), int(a.down_domain)}
        if not (pair & ACOUSTIC_DOMAINS) or not (pair & set(STRUCTURAL_DOMAINS)):
            continue
        candidates = acoustic_edges.get(tag, [])
        if not candidates:
            missed.append(tag)
            continue
        vm = solid.edge_mid_nodes[tuple(sorted((int(va), int(vb))))]
        p0 = solid.points_rz_m[va]
        p1 = solid.points_rz_m[vb]
        tang = p1 - p0
        length = float(np.linalg.norm(tang))
        if length <= 0:
            continue
        n = np.array([tang[1], -tang[0]]) / length
        adjacent = solid_edge_tris.get(tuple(sorted((int(va), int(vb)))), [])
        if adjacent:
            solid_cent = solid_centroids[adjacent[0]]
            if np.dot(n, solid_cent - 0.5 * (p0 + p1)) < 0:
                n = -n

        for t, w in zip(ts, ws):
            x = (1.0 - t) * p0 + t * p1
            best = None
            for ga, gb in candidates:
                q0 = mesh.points_rz_m[ga]
                q1 = mesh.points_rz_m[gb]
                qv = q1 - q0
                tau = float(np.clip(np.dot(x - q0, qv) / max(np.dot(qv, qv), 1e-30), 0.0, 1.0))
                distance = float(np.linalg.norm(x - (q0 + tau * qv)))
                if best is None or distance < best[0]:
                    best = (distance, ga, gb, tau)
            distance, ga, gb, tau = best
            max_projection_distance = max(max_projection_distance, distance)
            has_midpoint = (
                pressure_operator is not None
                and hasattr(pressure_operator, "pressure_dof_for_edge")
                and (
                    not hasattr(pressure_operator, "has_pressure_dof_for_edge")
                    or pressure_operator.has_pressure_dof_for_edge(ga, gb)
                )
            )
            if has_midpoint:
                pcols = [
                    ac_model.acoustic_node_map[ga],
                    ac_model.acoustic_node_map[gb],
                    pressure_operator.pressure_dof_for_edge(ga, gb),
                ]
                Np = edge_shapes(tau)
            else:
                pcols = [ac_model.acoustic_node_map[ga], ac_model.acoustic_node_map[gb]]
                Np = np.array([1.0 - tau, tau])
            Ns = edge_shapes(float(t))
            wt = 2 * math.pi * max(float(x[0]), 1e-12) * length * w
            for i, node in enumerate((va, vb, vm)):
                for j, pc in enumerate(pcols):
                    value = wt * Ns[i] * Np[j]
                    rows += [2 * node, 2 * node + 1]
                    cols += [pc, pc]
                    vals += [n[0] * value, n[1] * value]
        used_tags.append(tag)

    pressure_dofs = pressure_operator.n2 if pressure_operator is not None else len(ac_model.acoustic_nodes_global)
    G = coo_matrix((vals, (rows, cols)), shape=(solid.ndof, pressure_dofs)).tocsr()
    return G, {
        "G_shape": list(G.shape),
        "interface_boundaries": sorted(set(map(int, used_tags))),
        "missed_boundary_tags": sorted(set(map(int, missed))),
        "quadrature_order": 4,
        "pressure_trace_order": "mixed_P1_P2" if pressure_operator is not None else 1,
        "mesh_coupling": "nonconforming_closest_same_boundary",
        "max_projection_distance_m": float(max_projection_distance),
    }


class P2BoundarySampler:
    def __init__(self, solid: P2SolidModel, points):
        self.solid = solid
        self.points = points.reset_index(drop=True)
        by_tag: dict[int, list[tuple[int, int]]] = {}
        for a, b, tag in solid.boundary_edges:
            by_tag.setdefault(int(tag), []).append((int(a), int(b)))
        v0 = []
        v1 = []
        vm = []
        tt = []
        dd = []
        for row in self.points.itertuples(index=False):
            x = np.array([float(row.r_m), float(row.z_m)])
            best = None
            for a, b in by_tag.get(int(row.boundary_id), []):
                p0 = solid.points_rz_m[a]
                p1 = solid.points_rz_m[b]
                v = p1 - p0
                t = float(np.clip(np.dot(x - p0, v) / max(np.dot(v, v), 1e-30), 0.0, 1.0))
                d = float(np.linalg.norm(x - (p0 + t * v)))
                if best is None or d < best[0]:
                    best = (d, a, b, t)
            if best is None:
                raise RuntimeError(f"no P2 structural edge for boundary {row.boundary_id}")
            d, a, b, t = best
            v0.append(a)
            v1.append(b)
            vm.append(solid.edge_mid_nodes[tuple(sorted((a, b)))])
            tt.append(t)
            dd.append(d)
        self.v0 = np.asarray(v0, int)
        self.v1 = np.asarray(v1, int)
        self.vm = np.asarray(vm, int)
        self.t = np.asarray(tt, float)
        self.dist = np.asarray(dd, float)

    def sample(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.t
        N0 = (1 - t) * (1 - 2 * t)
        N1 = t * (2 * t - 1)
        Nm = 4 * t * (1 - t)
        ur = N0 * u[2 * self.v0] + N1 * u[2 * self.v1] + Nm * u[2 * self.vm]
        uz = N0 * u[2 * self.v0 + 1] + N1 * u[2 * self.v1 + 1] + Nm * u[2 * self.vm + 1]
        if {"normal_r", "normal_z"}.issubset(self.points.columns):
            un = ur * self.points.normal_r.to_numpy(float) + uz * self.points.normal_z.to_numpy(float)
        else:
            un = np.zeros_like(ur)
        return ur, uz, un
