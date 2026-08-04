from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from loudspeaker_axisym_fem.narrow_region_acoustics import equivalent_narrow_region_coefficients
from loudspeaker_axisym_fem.stage4C_acoustic_structure import (
    ACOUSTIC_DOMAINS,
    NRA_DOMAINS,
    PML_DOMAINS,
    AcousticStructureModel,
    _tri_area_grads,
)
from loudspeaker_axisym_fem.stage4D_exterior_nra import HKBoundaryInfo

from p2_pml_operator import (
    C0_COMSOL,
    duffy_rule,
    exact_stretched_points,
    p2_shape_and_grad,
)


class GlobalP2AcousticOperator:
    """Conforming P2 pressure discretization in physical air, NRA and PML domains."""

    def __init__(
        self,
        model: AcousticStructureModel,
        coefficient_csv=None,
        *,
        c0_m_s: float = C0_COMSOL,
        rho0_kg_m3: float = 1.2041,
        quadrature_order: int = 4,
    ):
        if coefficient_csv is not None:
            raise ValueError("calibrated NRA tables are disabled")
        self.model = model
        self.mesh = model.mesh
        self.n = len(model.acoustic_nodes_global)
        self.c0 = float(c0_m_s)
        self.rho0 = float(rho0_kg_m3)
        self.quadrature_order = int(quadrature_order)
        self._build_topology()
        self._physical_by_domain = self._assemble_physical_domains()
        self._pml_cache: dict[tuple[float, int], tuple[csr_matrix, csr_matrix, dict]] = {}

    def _build_topology(self):
        amap = self.model.acoustic_node_map
        edges: set[tuple[int, int]] = set()
        records = []
        for it, (tri, dom) in enumerate(zip(self.mesh.triangles, self.mesh.tri_domains.astype(int))):
            dom = int(dom)
            if dom not in ACOUSTIC_DOMAINS:
                continue
            es = [
                tuple(sorted((int(tri[0]), int(tri[1])))),
                tuple(sorted((int(tri[1]), int(tri[2])))),
                tuple(sorted((int(tri[2]), int(tri[0])))),
            ]
            edges.update(es)
            xy = self.mesh.points_rz_m[tri]
            area, grads = _tri_area_grads(xy)
            records.append({
                "triangle_id": it,
                "domain": dom,
                "global_vertices": np.asarray(tri, int),
                "vertex_dofs": np.asarray([amap[int(g)] for g in tri], int),
                "xy": xy,
                "area": float(area),
                "grads": grads,
                "edges": es,
            })
        self.edge_dof = {e: self.n + i for i, e in enumerate(sorted(edges))}
        self.edge_point = {
            e: 0.5 * (self.mesh.points_rz_m[e[0]] + self.mesh.points_rz_m[e[1]])
            for e in self.edge_dof
        }
        self.n2 = self.n + len(self.edge_dof)
        for rec in records:
            rec["dofs6"] = np.r_[rec["vertex_dofs"], [self.edge_dof[e] for e in rec["edges"]]]
        self.triangles = records
        self.triangle_by_id = {int(rec["triangle_id"]): rec for rec in records}

    def _assemble_physical_domains(self):
        Nq, Wq = duffy_rule(self.quadrature_order)
        rows: dict[int, list[int]] = {}
        cols: dict[int, list[int]] = {}
        kvals: dict[int, list[float]] = {}
        mvals: dict[int, list[float]] = {}
        for rec in self.triangles:
            dom = rec["domain"]
            if dom in PML_DOMAINS:
                continue
            rows.setdefault(dom, []); cols.setdefault(dom, [])
            kvals.setdefault(dom, []); mvals.setdefault(dom, [])
            Ke = np.zeros((6, 6), float)
            Me = np.zeros((6, 6), float)
            for N3, wref in zip(Nq, Wq):
                N6, g6 = p2_shape_and_grad(N3, rec["grads"])
                x = N3 @ rec["xy"]
                wt = 2 * rec["area"] * wref * 2 * math.pi * max(float(x[0]), 1e-15)
                Ke += wt * (g6 @ g6.T)
                Me += wt * np.outer(N6, N6)
            dofs = rec["dofs6"]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows[dom] += rr.ravel().tolist()
            cols[dom] += cc.ravel().tolist()
            kvals[dom] += Ke.ravel().tolist()
            mvals[dom] += Me.ravel().tolist()
        return {
            dom: (
                coo_matrix((kvals[dom], (rows[dom], cols[dom])), shape=(self.n2, self.n2)).tocsr(),
                coo_matrix((mvals[dom], (rows[dom], cols[dom])), shape=(self.n2, self.n2)).tocsr(),
            )
            for dom in rows
        }

    def _assemble_pml(self, freq_Hz: float, qorder: int):
        key = (float(freq_Hz), int(qorder))
        if key in self._pml_cache:
            return self._pml_cache[key]
        wavelength = self.c0 / float(freq_Hz)
        Nq, Wq = duffy_rule(qorder)
        rr: list[int] = []; cc: list[int] = []
        kv: list[complex] = []; mv: list[complex] = []
        det_stats = []; volume_stats = []
        for rec in self.triangles:
            if rec["domain"] not in PML_DOMAINS:
                continue
            xy = rec["xy"]
            pts6 = np.vstack([
                xy,
                0.5 * (xy[0] + xy[1]),
                0.5 * (xy[1] + xy[2]),
                0.5 * (xy[2] + xy[0]),
            ])
            stretched6 = exact_stretched_points(pts6, wavelength)
            Ke = np.zeros((6, 6), complex)
            Me = np.zeros((6, 6), complex)
            for N3, wref in zip(Nq, Wq):
                N6, g6 = p2_shape_and_grad(N3, rec["grads"])
                x = N3 @ xy
                xt = N6 @ stretched6
                J = stretched6.T @ g6
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                axisym_ratio = xt[0] / max(float(x[0]), 1e-15)
                volume = detJ * axisym_ratio
                tensor = volume * (invJ @ invJ.T)
                wt = 2 * rec["area"] * wref * 2 * math.pi * max(float(x[0]), 1e-15)
                Ke += wt * (g6 @ tensor @ g6.T)
                Me += wt * volume * np.outer(N6, N6)
                det_stats.append(abs(detJ)); volume_stats.append(abs(volume))
            dofs = rec["dofs6"]
            rloc, cloc = np.meshgrid(dofs, dofs, indexing="ij")
            rr += rloc.ravel().tolist(); cc += cloc.ravel().tolist()
            kv += Ke.ravel().tolist(); mv += Me.ravel().tolist()
        K = coo_matrix((kv, (rr, cc)), shape=(self.n2, self.n2)).tocsr()
        M = coo_matrix((mv, (rr, cc)), shape=(self.n2, self.n2)).tocsr()
        info = {
            "wavelength_m": wavelength,
            "quadrature_order": int(qorder),
            "pressure_order": 2,
            "base_pressure_dofs": self.n,
            "edge_pressure_dofs": len(self.edge_dof),
            "total_pressure_dofs": self.n2,
            "pml_triangles": int(sum(r["domain"] in PML_DOMAINS for r in self.triangles)),
            "detJ_abs_min": float(np.min(det_stats)),
            "detJ_abs_max": float(np.max(det_stats)),
            "axisym_volume_abs_min": float(np.min(volume_stats)),
            "axisym_volume_abs_max": float(np.max(volume_stats)),
        }
        self._pml_cache[key] = (K, M, info)
        return K, M, info

    def matrix(self, freq_Hz: float, qorder: int | None = None, *, nra_enabled: bool = True):
        qorder = int(qorder or self.quadrature_order)
        Kp, Mp, info = self._assemble_pml(freq_Hz, qorder)
        k = 2 * math.pi * float(freq_Hz) / self.c0
        A = Kp - k * k * Mp
        nra_info = {}
        for dom, (K, M) in self._physical_by_domain.items():
            sf = 1.0 + 0j
            mf = 1.0 + 0j
            if nra_enabled and dom in NRA_DOMAINS:
                coeff = equivalent_narrow_region_coefficients(
                    float(freq_Hz), NRA_DOMAINS[dom], rho0=self.rho0, c0=self.c0
                )
                sf = complex(coeff.stiffness_factor)
                mf = complex(coeff.mass_factor)
                nra_info[str(dom)] = {
                    "stiffness_factor": [sf.real, sf.imag],
                    "mass_factor": [mf.real, mf.imag],
                }
            A += sf * K - k * k * mf * M
        info = dict(info)
        info["nra_enabled"] = bool(nra_enabled)
        info["nra_model"] = "native_parallel_plate_thermoviscous" if nra_enabled else "lossless_pressure_acoustics"
        info["nra_factors"] = nra_info
        return A.tocsr(), info

    def base_pressure(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p[: self.n])

    def pressure_dof_for_edge(self, ga: int, gb: int) -> int:
        return self.edge_dof[tuple(sorted((int(ga), int(gb))))]

    def boundary_samples(
        self,
        p: np.ndarray,
        *,
        boundary_id: int = 93,
        intorder: int = 4,
        force_radial_normals: bool = True,
    ):
        """Evaluate the P2 pressure and its exact element gradient on a boundary."""
        xg, wg = np.polynomial.legendre.leggauss(int(intorder))
        ts = 0.5 * (xg + 1.0)
        ws = 0.5 * wg
        edge_to_tri: dict[tuple[int, int], list[int]] = {}
        for rec in self.triangles:
            for edge in rec["edges"]:
                edge_to_tri.setdefault(edge, []).append(int(rec["triangle_id"]))
        centers = self.mesh.points_rz_m[self.mesh.triangles].mean(axis=1)
        rs = []; zs = []; nr = []; nz = []; ds = []; pb = []; dpdn = []
        nseg = 0
        for seg, tag in zip(self.mesh.line_cells, self.mesh.line_tags):
            if int(tag) != int(boundary_id):
                continue
            adj = self.model.boundary_adjacency.get(int(tag))
            if adj is None:
                continue
            key = tuple(sorted(map(int, seg)))
            physical = [
                d for d in (adj.up_domain, adj.down_domain)
                if d in ACOUSTIC_DOMAINS and d not in PML_DOMAINS
            ]
            tri_id = None
            other_id = None
            for it in edge_to_tri.get(key, []):
                dom = int(self.mesh.tri_domains[it])
                if physical and dom == int(physical[0]):
                    tri_id = it
                elif dom in PML_DOMAINS:
                    other_id = it
            if tri_id is None:
                continue
            rec = self.triangle_by_id[int(tri_id)]
            p0 = self.mesh.points_rz_m[int(seg[0])]
            p1 = self.mesh.points_rz_m[int(seg[1])]
            tangent = p1 - p0
            length = float(np.linalg.norm(tangent))
            if length <= 0:
                continue
            normal = np.array([tangent[1], -tangent[0]], float) / length
            direction = (
                centers[other_id] - centers[tri_id]
                if other_id is not None
                else 0.5 * (p0 + p1) - centers[tri_id]
            )
            if np.dot(normal, direction) < 0:
                normal = -normal
            bary_matrix = np.vstack([rec["xy"].T, np.ones(3)])
            p6 = np.asarray(p[rec["dofs6"]], complex)
            for t, w in zip(ts, ws):
                x = (1.0 - t) * p0 + t * p1
                N3 = np.linalg.solve(bary_matrix, np.r_[x, 1.0])
                N6, g6 = p2_shape_and_grad(N3, rec["grads"])
                n_eval = normal
                if force_radial_normals:
                    n_eval = x / max(float(np.linalg.norm(x)), 1e-14)
                pressure = complex(N6 @ p6)
                gradient = p6 @ g6
                rs.append(float(x[0])); zs.append(float(x[1]))
                nr.append(float(n_eval[0])); nz.append(float(n_eval[1]))
                ds.append(float(length * w)); pb.append(pressure)
                dpdn.append(complex(gradient @ n_eval))
            nseg += 1
        if not rs:
            raise RuntimeError(f"No usable P2 Boundary {boundary_id} samples")
        return (
            np.asarray(rs), np.asarray(zs), np.asarray(nr), np.asarray(nz),
            np.asarray(ds), np.asarray(pb), np.asarray(dpdn),
        ), HKBoundaryInfo(
            n_samples=len(rs),
            boundary_id=int(boundary_id),
            source_segments=nseg,
            mean_radius_m=float(np.mean(rs)),
            mean_abs_pressure_Pa=float(np.mean(np.abs(pb))),
            mean_abs_dpdn_Pa_per_m=float(np.mean(np.abs(dpdn))),
        )

    def mixed_points_and_cells(self):
        base = self.mesh.points_rz_m[self.model.acoustic_nodes_global]
        edges = sorted(self.edge_dof, key=self.edge_dof.get)
        points = np.vstack([base, np.asarray([self.edge_point[e] for e in edges], float)])
        triangle6 = [rec["dofs6"] for rec in self.triangles]
        return points, np.zeros((0, 3), int), np.asarray(triangle6, int)

    def mixed_triangle6_domains(self) -> np.ndarray:
        """Domain IDs corresponding to the triangle6 block returned above."""
        return np.asarray([int(rec["domain"]) for rec in self.triangles], int)

    def pressure_for_mixed_points(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p[: self.n2])
