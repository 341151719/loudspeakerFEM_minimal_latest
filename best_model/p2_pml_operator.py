from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.spatial import cKDTree

from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    AcousticStructureModel,
    ACOUSTIC_DOMAINS,
    PML_DOMAINS,
    NRA_DOMAINS,
    _tri_area_grads,
)
from loudspeaker_axisym_fem.narrow_region_acoustics import equivalent_narrow_region_coefficients

C0_COMSOL = 343.203523929095
R0_PML = 0.165
PML_THICKNESS = 0.015


def duffy_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(order))
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    N = []
    W = []
    for ui, wi in zip(u, wu):
        for vi, wj in zip(u, wu):
            xi = ui
            eta = (1 - ui) * vi
            N.append([1 - xi - eta, xi, eta])
            W.append(wi * wj * (1 - ui))
    return np.asarray(N, float), np.asarray(W, float)


def p2_shape_and_grad(N3: np.ndarray, grads3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L1, L2, L3 = N3
    g1, g2, g3 = grads3
    N = np.array([
        L1 * (2 * L1 - 1),
        L2 * (2 * L2 - 1),
        L3 * (2 * L3 - 1),
        4 * L1 * L2,
        4 * L2 * L3,
        4 * L3 * L1,
    ])
    g = np.vstack([
        (4 * L1 - 1) * g1,
        (4 * L2 - 1) * g2,
        (4 * L3 - 1) * g3,
        4 * (L1 * g2 + L2 * g1),
        4 * (L2 * g3 + L3 * g2),
        4 * (L3 * g1 + L1 * g3),
    ])
    return N, g


def exact_stretched_points(points: np.ndarray, wavelength_m: float) -> np.ndarray:
    radius = np.hypot(points[:, 0], points[:, 1])
    direction = points / np.maximum(radius[:, None], 1e-15)
    xi = np.clip((radius - R0_PML) / PML_THICKNESS, 0.0, 1.0)
    mapped_radius = R0_PML + wavelength_m * xi**3 * (1.0 - 1.0j)
    return direction * mapped_radius[:, None]


def _embed(matrix: csr_matrix, total: int) -> csr_matrix:
    n = matrix.shape[0]
    extra = total - n
    if extra <= 0:
        return matrix.tocsr()
    return bmat([
        [matrix, csr_matrix((n, extra))],
        [csr_matrix((extra, n)), csr_matrix((extra, extra))],
    ], format="csr")


@dataclass
class PMLTopology:
    base_pressure_dofs: int
    extra_enriched_edge_dofs: int
    total_pressure_dofs: int
    enriched_triangles: int
    interface_constrained_edges: int


class LocalP2PMLOperator:
    """Selective P2 pressure operator with a spherical polynomial PML.

    PML triangles and the requested physical domains receive P2 pressure
    enrichment. Mid-edge DOFs at P1/P2 and physical/PML interfaces are
    constrained to the shared P1 trace, preserving a conforming mixed-order
    discretization and the COMSOL Boundary 93 exterior trace.
    """

    def __init__(
        self,
        model: AcousticStructureModel,
        coefficient_csv: str | Path | None = None,
        *,
        c0_m_s: float = C0_COMSOL,
        rho0_kg_m3: float = 1.2041,
        quadrature_order: int = 4,
        physical_p2_domains: tuple[int, ...] | list[int] = (),
    ):
        self.model = model
        self.mesh = model.mesh
        self.n = len(model.acoustic_nodes_global)
        self.c0 = float(c0_m_s)
        self.rho0 = float(rho0_kg_m3)
        self.quadrature_order = int(quadrature_order)
        self.physical_p2_domains = set(map(int, physical_p2_domains))
        self.enriched_domains = set(PML_DOMAINS) | self.physical_p2_domains
        if coefficient_csv is not None:
            raise ValueError(
                "calibrated NRA coefficient tables are disabled; use the native "
                "parallel-plate thermoviscous model"
            )
        self.K_by_domain, self.M_by_domain = self._assemble_base_domain_matrices()
        self.K_nonpml = sum((self.K_by_domain[d] for d in self.K_by_domain if d not in self.enriched_domains), csr_matrix((self.n, self.n)))
        self.M_nonpml = sum((self.M_by_domain[d] for d in self.M_by_domain if d not in self.enriched_domains), csr_matrix((self.n, self.n)))
        self._build_topology()
        self._cache: dict[tuple[float, int], tuple[csr_matrix, csr_matrix, dict]] = {}

    def _assemble_base_domain_matrices(self):
        amap = self.model.acoustic_node_map
        rowsK: dict[int, list[int]] = {}
        colsK: dict[int, list[int]] = {}
        valsK: dict[int, list[float]] = {}
        rowsM: dict[int, list[int]] = {}
        colsM: dict[int, list[int]] = {}
        valsM: dict[int, list[float]] = {}
        for tri, dom in zip(self.mesh.triangles, self.mesh.tri_domains.astype(int)):
            dom = int(dom)
            if dom not in ACOUSTIC_DOMAINS:
                continue
            xy = self.mesh.points_rz_m[tri]
            area, grads = _tri_area_grads(xy)
            if area <= 0 or grads is None:
                continue
            rbar = max(float(xy[:, 0].mean()), 1e-12)
            wt = 2 * math.pi * rbar * area
            Ke = wt * (grads @ grads.T)
            Me = (wt / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], float)
            loc = [amap[int(g)] for g in tri]
            rowsK.setdefault(dom, []); colsK.setdefault(dom, []); valsK.setdefault(dom, [])
            rowsM.setdefault(dom, []); colsM.setdefault(dom, []); valsM.setdefault(dom, [])
            for i in range(3):
                for j in range(3):
                    rowsK[dom].append(loc[i]); colsK[dom].append(loc[j]); valsK[dom].append(float(Ke[i, j]))
                    rowsM[dom].append(loc[i]); colsM[dom].append(loc[j]); valsM[dom].append(float(Me[i, j]))
        K = {d: coo_matrix((valsK[d], (rowsK[d], colsK[d])), shape=(self.n, self.n)).tocsr() for d in rowsK}
        M = {d: coo_matrix((valsM[d], (rowsM[d], colsM[d])), shape=(self.n, self.n)).tocsr() for d in rowsM}
        return K, M

    def _build_topology(self):
        amap = self.model.acoustic_node_map
        edge_domains: dict[tuple[int, int], set[int]] = {}
        all_enriched_edges: set[tuple[int, int]] = set()
        enriched_triangles = []
        for it, (tri, dom) in enumerate(zip(self.mesh.triangles, self.mesh.tri_domains.astype(int))):
            dom = int(dom)
            if dom not in ACOUSTIC_DOMAINS:
                continue
            edges = [
                tuple(sorted((int(tri[0]), int(tri[1])))),
                tuple(sorted((int(tri[1]), int(tri[2])))),
                tuple(sorted((int(tri[2]), int(tri[0])))),
            ]
            for e in edges:
                edge_domains.setdefault(e, set()).add(dom)
            if dom in self.enriched_domains:
                all_enriched_edges.update(edges)
                xy = self.mesh.points_rz_m[tri]
                area, grads = _tri_area_grads(xy)
                enriched_triangles.append({
                    "triangle_id": it,
                    "domain": dom,
                    "global_vertices": np.asarray(tri, int),
                    "xy": xy,
                    "area": float(area),
                    "grads": grads,
                    "base_dofs": np.asarray([amap[int(g)] for g in tri], int),
                    "edges": edges,
                })
        interface = set()
        for e in all_enriched_edges:
            ds = edge_domains[e]
            crosses_pml = any(d in PML_DOMAINS for d in ds) and any(
                d in ACOUSTIC_DOMAINS and d not in PML_DOMAINS for d in ds
            )
            crosses_order = any(d in self.enriched_domains for d in ds) and any(
                d in ACOUSTIC_DOMAINS and d not in self.enriched_domains for d in ds
            )
            if crosses_pml or crosses_order:
                interface.add(e)
        free_edges = sorted(all_enriched_edges - interface)
        self.edge_dof = {e: self.n + i for i, e in enumerate(free_edges)}
        self.edge_point = {e: 0.5 * (self.mesh.points_rz_m[e[0]] + self.mesh.points_rz_m[e[1]]) for e in free_edges}
        self.n2 = self.n + len(free_edges)
        for T in enriched_triangles:
            expansion = [[(int(T["base_dofs"][i]), 1.0)] for i in range(3)]
            for e in T["edges"]:
                if e in interface:
                    expansion.append([(amap[e[0]], 0.5), (amap[e[1]], 0.5)])
                else:
                    expansion.append([(self.edge_dof[e], 1.0)])
            T["expansion"] = expansion
        self.pml_triangles = enriched_triangles
        self.interface_edges = interface
        self.topology = PMLTopology(self.n, len(free_edges), self.n2, len(enriched_triangles), len(interface))

    def _nra_factors(self, domain: int, freq_Hz: float) -> tuple[complex, complex]:
        h = NRA_DOMAINS[int(domain)]
        c = equivalent_narrow_region_coefficients(
            float(freq_Hz), h, rho0=self.rho0, c0=self.c0
        )
        return complex(c.stiffness_factor), complex(c.mass_factor)

    def _assemble_pml(self, freq_Hz: float, qorder: int, nra_enabled: bool = True):
        key = (float(freq_Hz), int(qorder), bool(nra_enabled))
        if key in self._cache:
            return self._cache[key]
        wavelength = self.c0 / float(freq_Hz)
        Nq, Wq = duffy_rule(qorder)
        rr: list[int] = []
        cc: list[int] = []
        kv: list[complex] = []
        mv: list[complex] = []
        det_stats = []
        volume_stats = []
        enriched_nra_info = {}
        for T in self.pml_triangles:
            xy = T["xy"]
            pts6 = np.vstack([xy, 0.5 * (xy[0] + xy[1]), 0.5 * (xy[1] + xy[2]), 0.5 * (xy[2] + xy[0])])
            is_pml = int(T["domain"]) in PML_DOMAINS
            stretched6 = exact_stretched_points(pts6, wavelength) if is_pml else pts6.astype(complex)
            Ke = np.zeros((6, 6), complex)
            Me = np.zeros((6, 6), complex)
            for N3, wref in zip(Nq, Wq):
                N6, g6 = p2_shape_and_grad(N3, T["grads"])
                x = N3 @ xy
                wt = 2 * T["area"] * wref * 2 * math.pi * max(float(x[0]), 1e-15)
                if is_pml:
                    xt = N6 @ stretched6
                    J = stretched6.T @ g6
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    axisym_ratio = xt[0] / max(float(x[0]), 1e-15)
                    volume = detJ * axisym_ratio
                    tensor = volume * (invJ @ invJ.T)
                    Ke += wt * (g6 @ tensor @ g6.T)
                    Me += wt * volume * np.outer(N6, N6)
                    det_stats.append(abs(detJ))
                    volume_stats.append(abs(volume))
                else:
                    Ke += wt * (g6 @ g6.T)
                    Me += wt * np.outer(N6, N6)
            if nra_enabled and int(T["domain"]) in NRA_DOMAINS:
                sf, mf = self._nra_factors(int(T["domain"]), float(freq_Hz))
                Ke *= sf
                Me *= mf
                enriched_nra_info[str(int(T["domain"]))] = {
                    "stiffness_factor": [float(sf.real), float(sf.imag)],
                    "mass_factor": [float(mf.real), float(mf.imag)],
                }
            exp = T["expansion"]
            for i in range(6):
                for j in range(6):
                    for gi, wi in exp[i]:
                        for gj, wj in exp[j]:
                            rr.append(gi); cc.append(gj); kv.append(wi * wj * Ke[i, j]); mv.append(wi * wj * Me[i, j])
        K = coo_matrix((kv, (rr, cc)), shape=(self.n2, self.n2)).tocsr()
        M = coo_matrix((mv, (rr, cc)), shape=(self.n2, self.n2)).tocsr()
        info = {
            "wavelength_m": wavelength,
            "quadrature_order": int(qorder),
            "base_pressure_dofs": self.n,
            "extra_enriched_edge_dofs": self.topology.extra_enriched_edge_dofs,
            "total_pressure_dofs": self.n2,
            "enriched_triangles": self.topology.enriched_triangles,
            "physical_p2_domains": sorted(self.physical_p2_domains),
            "interface_constrained_edges": self.topology.interface_constrained_edges,
            "detJ_abs_min": float(np.min(det_stats)) if det_stats else None,
            "detJ_abs_max": float(np.max(det_stats)) if det_stats else None,
            "axisym_volume_abs_min": float(np.min(volume_stats)) if volume_stats else None,
            "axisym_volume_abs_max": float(np.max(volume_stats)) if volume_stats else None,
            "enriched_nra_factors": enriched_nra_info,
        }
        self._cache[key] = (K, M, info)
        return K, M, info

    def matrix(self, freq_Hz: float, qorder: int | None = None, *, nra_enabled: bool = True) -> tuple[csr_matrix, dict]:
        qorder = int(qorder or self.quadrature_order)
        Kp, Mp, info = self._assemble_pml(freq_Hz, qorder, nra_enabled=nra_enabled)
        k = 2 * math.pi * float(freq_Hz) / self.c0
        A = Kp - k * k * Mp + _embed(self.K_nonpml, self.n2) - k * k * _embed(self.M_nonpml, self.n2)
        nra_info = {}
        if nra_enabled:
            for dom in NRA_DOMAINS:
                if int(dom) in self.physical_p2_domains:
                    continue
                sf, mf = self._nra_factors(int(dom), float(freq_Hz))
                A += _embed((sf - 1) * self.K_by_domain[int(dom)] - k * k * (mf - 1) * self.M_by_domain[int(dom)], self.n2)
                nra_info[str(dom)] = {
                    "stiffness_factor": [float(sf.real), float(sf.imag)],
                    "mass_factor": [float(mf.real), float(mf.imag)],
                }
        info = dict(info)
        info["nra_enabled"] = bool(nra_enabled)
        info["nra_model"] = "native_parallel_plate_thermoviscous" if nra_enabled else "lossless_pressure_acoustics"
        info["nra_factors"] = nra_info
        return A.tocsr(), info

    def has_pressure_dof_for_edge(self, ga: int, gb: int) -> bool:
        return tuple(sorted((int(ga), int(gb)))) in self.edge_dof

    def pressure_dof_for_edge(self, ga: int, gb: int) -> int:
        return self.edge_dof[tuple(sorted((int(ga), int(gb))))]

    def extend_base_vector(self, v: np.ndarray) -> np.ndarray:
        if len(v) != self.n:
            raise ValueError(f"expected base vector length {self.n}, got {len(v)}")
        return np.r_[v, np.zeros(self.n2 - self.n, dtype=v.dtype)]

    def base_pressure(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p[: self.n])

    def mixed_points_and_cells(self):
        base_points = self.mesh.points_rz_m[self.model.acoustic_nodes_global]
        extra_edges = sorted(self.edge_dof, key=self.edge_dof.get)
        extra_points = np.asarray([self.edge_point[e] for e in extra_edges], float) if extra_edges else np.zeros((0, 2))
        points = np.vstack([base_points, extra_points])
        physical = []
        pml6 = []
        amap = self.model.acoustic_node_map
        for tri, dom in zip(self.mesh.triangles, self.mesh.tri_domains.astype(int)):
            if int(dom) not in ACOUSTIC_DOMAINS:
                continue
            if int(dom) not in self.enriched_domains:
                physical.append([amap[int(g)] for g in tri])
        for T in self.pml_triangles:
            conn = list(map(int, T["base_dofs"]))
            for e in T["edges"]:
                if e in self.interface_edges:
                    # A constrained midpoint has no independent point. For VTK create a
                    # geometric duplicate appended after the independent mixed points.
                    conn.append(-1)
                else:
                    conn.append(self.edge_dof[e])
            pml6.append(conn)
        # Add interface midpoint visualization points and update negative indices.
        pml6_arr = np.asarray(pml6, int)
        if np.any(pml6_arr < 0):
            p = points.tolist()
            for i, T in enumerate(self.pml_triangles):
                for j, e in enumerate(T["edges"]):
                    if pml6_arr[i, 3 + j] < 0:
                        pml6_arr[i, 3 + j] = len(p)
                        p.append((0.5 * (self.mesh.points_rz_m[e[0]] + self.mesh.points_rz_m[e[1]])).tolist())
            points = np.asarray(p, float)
        return points, np.asarray(physical, int), pml6_arr

    def mixed_triangle6_domains(self) -> np.ndarray:
        """Domain IDs corresponding to the triangle6 block returned above."""
        return np.asarray([int(T["domain"]) for T in self.pml_triangles], int)

    def pressure_for_mixed_points(self, p: np.ndarray) -> np.ndarray:
        points, _, pml6 = self.mixed_points_and_cells()
        vals = np.zeros(len(points), complex)
        vals[: self.n2] = p[: self.n2]
        # Constrained interface visualization points appear after n2. Reconstruct by
        # locating the matching edge midpoint in each PML triangle.
        for T, conn in zip(self.pml_triangles, pml6):
            for j, e in enumerate(T["edges"]):
                idx = int(conn[3 + j])
                if idx >= self.n2:
                    vals[idx] = 0.5 * (p[self.model.acoustic_node_map[e[0]]] + p[self.model.acoustic_node_map[e[1]]])
        return vals
