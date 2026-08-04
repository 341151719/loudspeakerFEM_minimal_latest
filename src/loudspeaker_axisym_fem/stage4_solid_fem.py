from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh, splu

from .axisym_magnetics import TaggedTriMesh, load_tagged_meshio

STRUCTURAL_DOMAINS = (3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25)
COIL_DOMAINS = (17, 18, 19)
FIXED_BOUNDARIES = (81, 85)


@dataclass(frozen=True)
class SolidMaterial:
    E: float
    nu: float
    rho: float
    loss_factor: float = 0.0
    beta_dK: float = 0.0
    label: str = ""


def default_stage4_materials(omega_loss: float = 2.0 * math.pi * 40.0) -> dict[int, SolidMaterial]:
    mats: dict[int, SolidMaterial] = {}
    # COMSOL tutorial material cards as exported in the .m file.
    for d in (3, 21):
        mats[d] = SolidMaterial(2.0e9, 0.42, 1200.0, loss_factor=0.04, label="Composite")
    for d in (9, 10, 11, 12, 13, 14, 15, 16):
        mats[d] = SolidMaterial(70.0e9, 0.33, 2000.0, loss_factor=0.04, label="Glass Fiber")
    for d in (17, 18, 19):
        mats[d] = SolidMaterial(110.0e9, 0.35, 4500.0, loss_factor=0.05, label="Coil")
    mats[20] = SolidMaterial(0.58e9, 0.30, 650.0, beta_dK=0.14 / omega_loss, label="Cloth")
    mats[25] = SolidMaterial(5.0e6, 0.40, 67.0, beta_dK=0.46 / omega_loss, label="Foam")
    return mats


@dataclass
class SolidFEMModel:
    points_rz_m: np.ndarray
    triangles: np.ndarray
    domains: np.ndarray
    global_node_ids: np.ndarray
    fixed_nodes_local: np.ndarray
    K_by_domain: dict[int, csr_matrix]
    M: csr_matrix
    load_unit_z_N: np.ndarray
    coil_volume_m3: float
    total_structural_volume_m3: float
    free_dofs: np.ndarray
    material_summary: dict[str, dict]

    @property
    def ndof(self) -> int:
        return int(self.points_rz_m.shape[0] * 2)

    @property
    def K_real(self) -> csr_matrix:
        K = None
        for v in self.K_by_domain.values():
            K = v.copy() if K is None else K + v
        return K.tocsr() if K is not None else csr_matrix((self.ndof, self.ndof))

    def summary(self) -> dict:
        return {
            "n_structural_nodes": int(self.points_rz_m.shape[0]),
            "n_structural_triangles": int(self.triangles.shape[0]),
            "ndof_total": self.ndof,
            "ndof_free": int(len(self.free_dofs)),
            "fixed_nodes": int(len(self.fixed_nodes_local)),
            "coil_volume_m3": float(self.coil_volume_m3),
            "total_structural_volume_m3": float(self.total_structural_volume_m3),
            "domains": sorted(int(x) for x in set(self.domains.tolist())),
            "materials": self.material_summary,
        }


def _tri_area_and_grads(p: np.ndarray):
    r0, z0 = p[0]
    r1, z1 = p[1]
    r2, z2 = p[2]
    twice = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0)
    area = 0.5 * abs(twice)
    if area <= 0:
        return 0.0, None
    # gradients of shape functions wrt r,z, with sign following orientation
    denom = twice
    grads = np.array([
        [(z1-z2)/denom, (r2-r1)/denom],
        [(z2-z0)/denom, (r0-r2)/denom],
        [(z0-z1)/denom, (r1-r0)/denom],
    ], dtype=float)
    return area, grads


def _elastic_D(E: float, nu: float) -> np.ndarray:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array([
        [lam + 2*mu, lam, lam, 0.0],
        [lam, lam + 2*mu, lam, 0.0],
        [lam, lam, lam + 2*mu, 0.0],
        [0.0, 0.0, 0.0, mu],
    ], dtype=float)



def _refine_structural_mesh(pts: np.ndarray, tris: np.ndarray, doms: np.ndarray, fixed_nodes: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly subdivide structural triangles while preserving domain ids and fixed edges.

    This is a lightweight replacement for COMSOL's quadratic solid elements in the
    Stage-4B matrix audit.  New mid-edge nodes are marked fixed when both parent edge
    endpoints are fixed; this is exact for the straight exported boundary segments.
    """
    pts = np.asarray(pts, dtype=float)
    tris = np.asarray(tris, dtype=int)
    doms = np.asarray(doms, dtype=int)
    fixed = set(int(x) for x in np.asarray(fixed_nodes, dtype=int).ravel())
    for _ in range(int(levels)):
        edge_mid: dict[tuple[int, int], int] = {}
        pts_list = [tuple(x) for x in pts]
        new_tris = []
        new_doms = []
        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid:
                return edge_mid[key]
            pa = pts[key[0]]; pb = pts[key[1]]
            idx = len(pts_list)
            pts_list.append(tuple(0.5*(pa+pb)))
            edge_mid[key] = idx
            if key[0] in fixed and key[1] in fixed:
                fixed.add(idx)
            return idx
        for tri, dom in zip(tris, doms):
            a,b,c = map(int, tri)
            ab = midpoint(a,b); bc = midpoint(b,c); ca = midpoint(c,a)
            new_tris.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            new_doms.extend([int(dom)]*4)
        pts = np.asarray(pts_list, dtype=float)
        tris = np.asarray(new_tris, dtype=int)
        doms = np.asarray(new_doms, dtype=int)
    return pts, tris, doms, np.asarray(sorted(fixed), dtype=int)


def build_stage4_solid_model(mesh: TaggedTriMesh | str | Path, *, materials: Mapping[int, SolidMaterial] | None = None, uniform_refine: int = 0) -> SolidFEMModel:
    if not isinstance(mesh, TaggedTriMesh):
        mesh = load_tagged_meshio(mesh)
    mats = dict(materials or default_stage4_materials())
    mask = np.isin(mesh.tri_domains, STRUCTURAL_DOMAINS)
    tris_g = mesh.triangles[mask]
    doms = mesh.tri_domains[mask].astype(int)
    used_nodes = np.unique(tris_g.ravel())
    g2l = {int(g): i for i, g in enumerate(used_nodes)}
    tris = np.vectorize(g2l.__getitem__)(tris_g).astype(int)
    pts = mesh.points_rz_m[used_nodes]
    fixed_g = mesh.boundary_nodes(FIXED_BOUNDARIES)
    fixed_l = np.array([g2l[int(x)] for x in fixed_g if int(x) in g2l], dtype=int)
    if int(uniform_refine) > 0:
        pts, tris, doms, fixed_l = _refine_structural_mesh(pts, tris, doms, fixed_l, int(uniform_refine))
        used_nodes = np.arange(len(pts), dtype=int)
    n = len(pts)
    ndof = 2*n
    rows_by_dom: dict[int, list[int]] = {d: [] for d in sorted(set(doms.tolist()))}
    cols_by_dom: dict[int, list[int]] = {d: [] for d in sorted(set(doms.tolist()))}
    vals_by_dom: dict[int, list[float]] = {d: [] for d in sorted(set(doms.tolist()))}
    m_rows: list[int] = []
    m_cols: list[int] = []
    m_vals: list[float] = []
    load = np.zeros(ndof, dtype=float)
    coil_volume = 0.0
    total_vol = 0.0

    for tri, dom in zip(tris, doms):
        p = pts[tri]
        area, grads = _tri_area_and_grads(p)
        if area <= 0 or grads is None:
            continue
        rbar = max(float(np.mean(p[:,0])), 1e-9)
        weight = 2.0 * math.pi * rbar * area
        mat = mats.get(int(dom))
        if mat is None:
            raise KeyError(f"missing material for structural domain {dom}")
        D = _elastic_D(mat.E, mat.nu)
        B = np.zeros((4, 6), dtype=float)
        for a in range(3):
            dNdr, dNdz = grads[a]
            N = 1.0/3.0
            ur = 2*a
            uz = 2*a + 1
            B[0, ur] = dNdr
            B[1, ur] = N / rbar
            B[2, uz] = dNdz
            B[3, ur] = dNdz
            B[3, uz] = dNdr
        Ke = weight * (B.T @ D @ B)
        MeN = (weight * mat.rho / 12.0) * np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
        dofs = []
        for a in tri:
            dofs.extend([2*int(a), 2*int(a)+1])
        for ia, I in enumerate(dofs):
            for ja, J in enumerate(dofs):
                rows_by_dom[int(dom)].append(I); cols_by_dom[int(dom)].append(J); vals_by_dom[int(dom)].append(float(Ke[ia,ja]))
        for a in range(3):
            for b in range(3):
                for comp in range(2):
                    I = 2*int(tri[a]) + comp
                    J = 2*int(tri[b]) + comp
                    m_rows.append(I); m_cols.append(J); m_vals.append(float(MeN[a,b]))
        total_vol += weight
        if int(dom) in COIL_DOMAINS:
            coil_volume += weight
            for a in tri:
                load[2*int(a)+1] += weight / 3.0

    if coil_volume <= 0:
        raise RuntimeError("coil volume is zero; domains 17-19 missing from structural mesh")
    load /= coil_volume

    K_by_domain: dict[int, csr_matrix] = {}
    for dom in rows_by_dom:
        K_by_domain[dom] = coo_matrix((vals_by_dom[dom], (rows_by_dom[dom], cols_by_dom[dom])), shape=(ndof, ndof)).tocsr()
    M = coo_matrix((m_vals, (m_rows, m_cols)), shape=(ndof, ndof)).tocsr()

    fixed_dofs = np.unique(np.concatenate([2*fixed_l, 2*fixed_l+1])) if fixed_l.size else np.array([], dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    mat_summary = {}
    for dom, mat in mats.items():
        if dom in set(doms.tolist()):
            mat_summary[str(dom)] = asdict(mat)
    return SolidFEMModel(pts, tris, doms, used_nodes, fixed_l, K_by_domain, M, load, coil_volume, total_vol, free, mat_summary)

def _complex_stiffness(model: SolidFEMModel, omega: float) -> csr_matrix:
    Kc = None
    for dom, Kd in model.K_by_domain.items():
        m = default_stage4_materials()[int(dom)]
        eta = m.loss_factor + omega * m.beta_dK
        block = Kd.astype(complex) * (1.0 + 1j*eta)
        Kc = block if Kc is None else Kc + block
    return Kc.tocsr() if Kc is not None else csr_matrix((model.ndof, model.ndof), dtype=complex)


def solve_structural_response(model: SolidFEMModel, freqs_Hz: Iterable[float], force_N: np.ndarray | None = None) -> dict[str, np.ndarray]:
    freqs = np.asarray(list(freqs_Hz), dtype=float)
    fvec = np.asarray(force_N if force_N is not None else model.load_unit_z_N, dtype=complex)
    free = model.free_dofs
    Mff = model.M[free][:, free].astype(complex)
    bf = fvec[free]
    disp = np.zeros((len(freqs), model.ndof), dtype=complex)
    coil_disp = np.zeros(len(freqs), dtype=complex)
    mech_compliance = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        w = 2.0 * math.pi * float(f)
        Kc = _complex_stiffness(model, w)
        Aff = (Kc[free][:, free] - (w*w) * Mff).tocsc()
        sol = splu(Aff).solve(bf)
        u = np.zeros(model.ndof, dtype=complex)
        u[free] = sol
        disp[i] = u
        # Since load_unit_z is normalized as unit force, q = f^T u is the force-weighted coil-average axial displacement.
        q = np.vdot(model.load_unit_z_N, u)  # load is real; vdot conjugates first argument only.
        coil_disp[i] = q
        mech_compliance[i] = q
    omega = 2.0 * math.pi * freqs
    v_per_N = 1j * omega * mech_compliance
    Zm = 1.0 / np.where(np.abs(v_per_N) > 1e-300, v_per_N, np.nan + 0j)
    return {
        "f_Hz": freqs,
        "displacement_per_N": disp,
        "coil_average_displacement_per_N_m": coil_disp,
        "velocity_per_N_m_s_per_N": v_per_N,
        "mechanical_impedance_N_s_m": Zm,
    }


def compute_eigenmodes(model: SolidFEMModel, *, nmodes: int = 10, sigma_Hz: float | None = None) -> dict[str, np.ndarray]:
    free = model.free_dofs
    K = model.K_real[free][:, free].tocsc()
    M = model.M[free][:, free].tocsc()
    if sigma_Hz is None:
        vals, vecs = eigsh(K, k=min(nmodes, K.shape[0]-2), M=M, which="SM")
    else:
        sigma = (2.0 * math.pi * sigma_Hz)**2
        vals, vecs = eigsh(K, k=min(nmodes, K.shape[0]-2), M=M, sigma=sigma, which="LM")
    vals = np.real(vals)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    freqs = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * math.pi)
    full = np.zeros((len(freqs), model.ndof), dtype=float)
    for i in range(len(freqs)):
        full[i, free] = vecs[:, i]
    return {"f_Hz": freqs, "modes": full, "lambda_rad2_s2": vals}


def modal_count_nodes(model: SolidFEMModel) -> dict:
    return model.summary()
