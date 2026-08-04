from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping
import math
import re

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import splu

from .axisym_magnetics import TaggedTriMesh, load_tagged_meshio
from .stage4_solid_fem import SolidFEMModel, build_stage4_solid_model, default_stage4_materials
from .stage4B_solid_electroacoustic import Stage4BSolidCouplingParameters
from .narrow_region_acoustics import equivalent_narrow_region_coefficients

STRUCTURAL_DOMAINS = set((3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25))
SOFT_IRON_DOMAINS = set((6, 23))
FERRITE_DOMAINS = set((24,))
# COMSOL ``Air`` selection is all domains minus material solids/iron/ferrite, so it includes PML 1 and 5.
ACOUSTIC_DOMAINS = set((1, 2, 4, 5, 7, 8, 22))
PML_DOMAINS = set((1, 5))
NRA_DOMAINS = {8: 0.4e-3, 22: 0.2e-3}
COIL_DOMAINS = set((17, 18, 19))
EXTERIOR_FIELD_BOUNDARY = 93


@dataclass
class BoundaryAdjacency:
    boundary_id: int
    v1: int
    v2: int
    up_domain: int
    down_domain: int
    curve_id: int


def parse_mphtxt_boundary_adjacency(mphtxt_path: str | Path) -> dict[int, BoundaryAdjacency]:
    """Parse COMSOL final-geometry boundary adjacency from exported ``mphtxt``.

    The line table has COMSOL boundary id order and columns ``v1 v2 ... up down curve``.
    Boundary ids are 1-based line indices.  This is enough to reconstruct which
    boundaries are acoustic-structure interfaces, which are external PML/air limits,
    and which line is Boundary 93 for exterior-field postprocessing.
    """
    lines = Path(mphtxt_path).read_text(errors='replace').splitlines()
    if '# Edges' not in lines:
        raise ValueError(f'{mphtxt_path} does not contain an # Edges table')
    n_edges = int(lines[lines.index('# Edges') - 1].split()[0])
    start = lines.index('# Edges') + 2
    out: dict[int, BoundaryAdjacency] = {}
    for i in range(n_edges):
        parts = lines[start + i].split()
        out[i + 1] = BoundaryAdjacency(
            boundary_id=i + 1,
            v1=int(parts[0]),
            v2=int(parts[1]),
            up_domain=int(parts[4]),
            down_domain=int(parts[5]),
            curve_id=int(parts[6]),
        )
    return out


@dataclass
class AcousticStructureModel:
    mesh: TaggedTriMesh
    solid: SolidFEMModel
    acoustic_nodes_global: np.ndarray
    acoustic_node_map: dict[int, int]
    pressure_free_dofs: np.ndarray
    pressure_dirichlet_dofs: np.ndarray
    Kp: csr_matrix
    Mp: csr_matrix
    Mp_pml: csr_matrix
    Mnra: dict[int, csr_matrix]
    Knra: dict[int, csr_matrix]
    G_sp: csr_matrix  # full solid dof x full pressure node matrix, ∫ Ns n Np dΓ
    interface_boundaries: list[int]
    exterior_boundary_facets: np.ndarray
    exterior_boundary_tags: np.ndarray
    boundary_adjacency: dict[int, BoundaryAdjacency]

    def summary(self) -> dict:
        return {
            'n_acoustic_nodes': int(len(self.acoustic_nodes_global)),
            'n_pressure_free_dofs': int(len(self.pressure_free_dofs)),
            'n_pressure_dirichlet_dofs': int(len(self.pressure_dirichlet_dofs)),
            'n_interface_boundaries': int(len(self.interface_boundaries)),
            'interface_boundaries': [int(x) for x in self.interface_boundaries],
            'exterior_field_boundary_present': bool(np.any(self.exterior_boundary_tags == EXTERIOR_FIELD_BOUNDARY)),
            'solid': self.solid.summary(),
        }


def _tri_area_grads(p: np.ndarray):
    r0, z0 = p[0]
    r1, z1 = p[1]
    r2, z2 = p[2]
    det = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0)
    area = 0.5*abs(det)
    if area <= 0:
        return 0.0, None
    grads = np.array([
        [(z1-z2)/det, (r2-r1)/det],
        [(z2-z0)/det, (r0-r2)/det],
        [(z0-z1)/det, (r1-r0)/det],
    ], dtype=float)
    return area, grads


def _acoustic_element_mats(points: np.ndarray, tri: np.ndarray, domain: int, *, c0: float, pml_strength: float = 0.35):
    p = points[tri]
    area, grads = _tri_area_grads(p)
    if area <= 0 or grads is None:
        return None
    rbar = max(float(np.mean(p[:, 0])), 1e-9)
    weight = 2.0*math.pi*rbar*area
    K = weight * (grads @ grads.T)
    M = (weight/12.0)*np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
    # The true COMSOL PML is a coordinate system; this scalar damping is a stable first
    # Stage-4C implementation preserving full acoustic-structure assembly semantics.
    pml = M if int(domain) in PML_DOMAINS else np.zeros((3,3), dtype=float)
    nra = M if int(domain) in NRA_DOMAINS else np.zeros((3,3), dtype=float)
    return K, M, pml, nra


def _edge_triangles(triangles: np.ndarray) -> dict[tuple[int,int], list[int]]:
    out: dict[tuple[int,int], list[int]] = {}
    for it, tri in enumerate(triangles):
        a,b,c = map(int, tri)
        for e in ((a,b),(b,c),(c,a)):
            key = e if e[0] < e[1] else (e[1], e[0])
            out.setdefault(key, []).append(it)
    return out


def _choose_normal_from_acoustic_to_solid(mesh: TaggedTriMesh, seg: np.ndarray, acoustic_dom: int, solid_dom: int) -> np.ndarray:
    p0 = mesh.points_rz_m[int(seg[0])]
    p1 = mesh.points_rz_m[int(seg[1])]
    t = p1 - p0
    L = float(np.linalg.norm(t))
    if L <= 0:
        return np.array([0.0, 1.0])
    n1 = np.array([t[1], -t[0]], dtype=float) / L
    # Use adjacent triangle centroids to orient the normal from acoustic side to structure.
    edge_map = _choose_normal_from_acoustic_to_solid._edge_map  # type: ignore[attr-defined]
    cents = _choose_normal_from_acoustic_to_solid._centroids  # type: ignore[attr-defined]
    doms = _choose_normal_from_acoustic_to_solid._domains  # type: ignore[attr-defined]
    key = tuple(sorted(map(int, seg)))
    ac = None; st = None
    for it in edge_map.get(key, []):
        d = int(doms[it])
        if d == int(acoustic_dom): ac = cents[it]
        if d == int(solid_dom): st = cents[it]
    if ac is not None and st is not None:
        v = st - ac
        if np.dot(n1, v) < 0:
            n1 = -n1
    else:
        # Fall back to pointing toward increasing z/r based on geometry center.
        mid = 0.5*(p0+p1)
        if np.dot(n1, np.array([0.02, -0.06]) - mid) < 0:
            n1 = -n1
    return n1


def build_stage4C_acoustic_structure_model(
    mesh: TaggedTriMesh | str | Path,
    mphtxt_path: str | Path,
    *,
    solid_uniform_refine: int = 0,
    c0: float = 343.0,
) -> AcousticStructureModel:
    if not isinstance(mesh, TaggedTriMesh):
        mesh = load_tagged_meshio(mesh)
    # IMPORTANT: ASB coupling requires shared interface nodes.  Therefore Stage-4C uses
    # the unrefined solid mesh by default.  Refined solid-only eigen checks remain Stage-4B.
    solid = build_stage4_solid_model(mesh, uniform_refine=solid_uniform_refine)
    if solid_uniform_refine != 0:
        raise ValueError('Stage 4C full ASB currently requires solid_uniform_refine=0 so acoustic and solid interface nodes match')
    adj = parse_mphtxt_boundary_adjacency(mphtxt_path)
    acoustic_mask = np.isin(mesh.tri_domains, list(ACOUSTIC_DOMAINS))
    acoustic_tris = mesh.triangles[acoustic_mask]
    acoustic_doms = mesh.tri_domains[acoustic_mask].astype(int)
    acoustic_nodes = np.unique(acoustic_tris.ravel())
    amap = {int(g): i for i, g in enumerate(acoustic_nodes)}
    npa = len(acoustic_nodes)
    rowsK=[]; colsK=[]; valsK=[]
    rowsM=[]; colsM=[]; valsM=[]
    rowsP=[]; colsP=[]; valsP=[]
    nra_rows = {8: [], 22: []}; nra_cols = {8: [], 22: []}; nra_vals = {8: [], 22: []}
    nraK_rows = {8: [], 22: []}; nraK_cols = {8: [], 22: []}; nraK_vals = {8: [], 22: []}
    for tri_g, dom in zip(acoustic_tris, acoustic_doms):
        tri_l = [amap[int(x)] for x in tri_g]
        mats = _acoustic_element_mats(mesh.points_rz_m, tri_g, int(dom), c0=c0)
        if mats is None: continue
        Ke, Me, Mpml, Mn = mats
        for a in range(3):
            for b in range(3):
                I=tri_l[a]; J=tri_l[b]
                rowsK.append(I); colsK.append(J); valsK.append(float(Ke[a,b]))
                rowsM.append(I); colsM.append(J); valsM.append(float(Me[a,b]))
                if Mpml[a,b] != 0:
                    rowsP.append(I); colsP.append(J); valsP.append(float(Mpml[a,b]))
                if int(dom) in NRA_DOMAINS:
                    if Mn[a,b] != 0:
                        nra_rows[int(dom)].append(I); nra_cols[int(dom)].append(J); nra_vals[int(dom)].append(float(Mn[a,b]))
                    if Ke[a,b] != 0:
                        nraK_rows[int(dom)].append(I); nraK_cols[int(dom)].append(J); nraK_vals[int(dom)].append(float(Ke[a,b]))
    Kp = coo_matrix((valsK,(rowsK,colsK)), shape=(npa,npa)).tocsr()
    Mp = coo_matrix((valsM,(rowsM,colsM)), shape=(npa,npa)).tocsr()
    Mpml = coo_matrix((valsP,(rowsP,colsP)), shape=(npa,npa)).tocsr()
    Mnra = {d: coo_matrix((nra_vals[d], (nra_rows[d], nra_cols[d])), shape=(npa,npa)).tocsr() for d in (8,22)}
    Knra = {d: coo_matrix((nraK_vals[d], (nraK_rows[d], nraK_cols[d])), shape=(npa,npa)).tocsr() for d in (8,22)}

    # Build solid global node -> local node map.  Stage-4C unrefined solid keeps global_node_ids from source mesh.
    sg2l = {int(g): i for i, g in enumerate(solid.global_node_ids)}
    Grows=[]; Gcols=[]; Gvals=[]
    interface_boundaries=[]
    edge_map = _edge_triangles(mesh.triangles)
    cents = mesh.points_rz_m[mesh.triangles].mean(axis=1)
    _choose_normal_from_acoustic_to_solid._edge_map = edge_map  # type: ignore[attr-defined]
    _choose_normal_from_acoustic_to_solid._centroids = cents  # type: ignore[attr-defined]
    _choose_normal_from_acoustic_to_solid._domains = mesh.tri_domains  # type: ignore[attr-defined]

    exterior_cells=[]; exterior_tags=[]
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        tag=int(tag)
        a=adj.get(tag)
        if a is None:
            continue
        dom_pair = {int(a.up_domain), int(a.down_domain)}
        acoustic_sides = list(dom_pair & ACOUSTIC_DOMAINS)
        solid_sides = list(dom_pair & STRUCTURAL_DOMAINS)
        if acoustic_sides and solid_sides:
            adom = acoustic_sides[0]; sdom = solid_sides[0]
            interface_boundaries.append(tag)
            nvec = _choose_normal_from_acoustic_to_solid(mesh, seg, adom, sdom)
            p0=mesh.points_rz_m[int(seg[0])]; p1=mesh.points_rz_m[int(seg[1])]
            L=float(np.linalg.norm(p1-p0)); rbar=max(float(np.mean([p0[0],p1[0]])),1e-9)
            W = 2.0*math.pi*rbar*L/6.0*np.array([[2,1],[1,2]], dtype=float)
            for ia, ga in enumerate(map(int, seg)):
                if ga not in sg2l or ga not in amap:
                    continue
                sl = sg2l[ga]; pl = amap[ga]
                for jb, gb in enumerate(map(int, seg)):
                    if gb not in amap:
                        continue
                    pj = amap[gb]
                    # pressure p_j contributes to solid dofs at structural node ga.
                    Grows.append(2*sl); Gcols.append(pj); Gvals.append(float(nvec[0]*W[ia,jb]))
                    Grows.append(2*sl+1); Gcols.append(pj); Gvals.append(float(nvec[1]*W[ia,jb]))
        if tag == EXTERIOR_FIELD_BOUNDARY:
            exterior_cells.append(seg.copy()); exterior_tags.append(tag)
    G = coo_matrix((Gvals,(Grows,Gcols)), shape=(solid.ndof,npa)).tocsr()

    # Dirichlet pressure on truncated exterior boundaries, excluding axis r=0 and excluding Boundary 93.
    dir_nodes=set()
    for seg, tag in zip(mesh.line_cells, mesh.line_tags):
        tag=int(tag)
        a=adj.get(tag)
        if a is None: continue
        dom_pair={int(a.up_domain), int(a.down_domain)}
        if 0 in dom_pair and (dom_pair & ACOUSTIC_DOMAINS) and tag != EXTERIOR_FIELD_BOUNDARY:
            rmean=float(np.mean(mesh.points_rz_m[seg,0]))
            if rmean > 1e-8:
                for g in map(int,seg):
                    if g in amap:
                        dir_nodes.add(amap[g])
    pressure_dir = np.asarray(sorted(dir_nodes), dtype=int)
    pressure_all = np.arange(npa, dtype=int)
    pressure_free = np.setdiff1d(pressure_all, pressure_dir)
    return AcousticStructureModel(
        mesh=mesh,
        solid=solid,
        acoustic_nodes_global=acoustic_nodes,
        acoustic_node_map=amap,
        pressure_free_dofs=pressure_free,
        pressure_dirichlet_dofs=pressure_dir,
        Kp=Kp,
        Mp=Mp,
        Mp_pml=Mpml,
        Mnra=Mnra,
        Knra=Knra,
        G_sp=G,
        interface_boundaries=sorted(set(interface_boundaries)),
        exterior_boundary_facets=np.asarray(exterior_cells, dtype=int) if exterior_cells else np.empty((0,2), dtype=int),
        exterior_boundary_tags=np.asarray(exterior_tags, dtype=int),
        boundary_adjacency=adj,
    )


def _complex_solid_stiffness(solid: SolidFEMModel, omega: float) -> csr_matrix:
    Kc = None
    mats = default_stage4_materials()
    for dom, Kd in solid.K_by_domain.items():
        mat = mats[int(dom)]
        eta = mat.loss_factor + omega*mat.beta_dK
        block = Kd.astype(complex)*(1.0 + 1j*eta)
        Kc = block if Kc is None else Kc + block
    if Kc is None:
        raise RuntimeError('empty solid stiffness')
    return Kc.tocsr()


def _acoustic_matrix(model: AcousticStructureModel, omega: float, *, rho0: float, c0: float, nra_enabled: bool = True) -> csr_matrix:
    k = omega/c0
    # Weak form: ∫ grad p grad q - k^2 p q, plus scalar complex PML damping.
    A = model.Kp.astype(complex) - (k*k)*model.Mp.astype(complex)
    if model.Mp_pml.nnz:
        # Damp PML domains with frequency-dependent lossy mass/stiffness surrogate.
        A = A + (0.75 - 1.8j)*(k*k)*model.Mp_pml.astype(complex)
    if nra_enabled:
        # Stage-4D: physically based slit thermoviscous equivalent coefficients
        # for COMSOL Narrow Region Acoustics domains 8 and 22.  The base global
        # matrix already includes K_nra - k0² M_nra, so add only the delta.
        f = omega/(2.0*math.pi)
        for dom, h in NRA_DOMAINS.items():
            if model.Mnra[dom].nnz:
                coeff = equivalent_narrow_region_coefficients(f, h, rho0=rho0, c0=c0)
                A = A + (coeff.stiffness_factor - 1.0) * model.Knra[dom].astype(complex)
                A = A - (k*k) * (coeff.mass_factor - 1.0) * model.Mnra[dom].astype(complex)
    return A.tocsr()


@dataclass
class Stage4CParameters:
    BL_N_A: float = 10.482177800
    V0_peak_V: float = 3.55
    rho0_kg_m3: float = 1.2041
    c0_m_s: float = 343.0
    p_ref_Pa: float = 20e-6
    observation_distance_m: float = 1.0
    radiation_radius_m: float = 0.070

    @property
    def Sd_m2(self) -> float:
        return math.pi*self.radiation_radius_m**2


def solve_coupled_unit_force(model: AcousticStructureModel, freqs_Hz: Iterable[float], params: Stage4CParameters, *, nra_enabled: bool = True) -> dict[str, np.ndarray]:
    f = np.asarray(list(freqs_Hz), dtype=float)
    solid = model.solid
    sf = solid.free_dofs
    pf = model.pressure_free_dofs
    Gsf = model.G_sp[sf][:, :].astype(complex)
    GT = model.G_sp.T[:, sf].astype(complex)
    bsolid_full = solid.load_unit_z_N.astype(complex)
    bsf = bsolid_full[sf]
    Msf = solid.M[sf][:, sf].astype(complex)
    disp = np.zeros((len(f), solid.ndof), dtype=complex)
    pfull = np.zeros((len(f), len(model.acoustic_nodes_global)), dtype=complex)
    q = np.zeros(len(f), dtype=complex)
    Zm = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        omega = 2.0*math.pi*float(fi)
        Ks = _complex_solid_stiffness(solid, omega)[sf][:, sf] - (omega*omega)*Msf
        Ap = _acoustic_matrix(model, omega, rho0=params.rho0_kg_m3, c0=params.c0_m_s, nra_enabled=nra_enabled)
        App = Ap[pf][:, pf]
        # Coupled ASB blocks.  Solid equation: S u - G p = F.
        # Acoustic equation: A p - rho*omega^2 G^T u = 0.
        Ablock = bmat([
            [Ks.tocsr(), -Gsf[:, pf]],
            [-params.rho0_kg_m3*omega*omega*GT[pf, :], App.tocsr()],
        ], format='csc')
        rhs = np.concatenate([bsf, np.zeros(len(pf), dtype=complex)])
        sol = splu(Ablock).solve(rhs)
        us = sol[:len(sf)]
        pp = sol[len(sf):]
        ufull = np.zeros(solid.ndof, dtype=complex); ufull[sf] = us
        pvec = np.zeros(len(model.acoustic_nodes_global), dtype=complex); pvec[pf] = pp
        disp[i] = ufull; pfull[i] = pvec
        q[i] = np.vdot(solid.load_unit_z_N, ufull)
        vperN = 1j*omega*q[i]
        Zm[i] = 1.0/vperN if abs(vperN) > 1e-300 else np.nan+0j
    return {
        'f_Hz': f,
        'displacement_per_N': disp,
        'pressure_per_N': pfull,
        'coil_average_displacement_per_N_m': q,
        'velocity_per_N_m_s_per_N': 1j*(2.0*np.pi*f)*q,
        'mechanical_impedance_N_s_m': Zm,
    }


def solve_stage4C_full_asb(freqs_Hz: Iterable[float], Zb_ohm: np.ndarray, model: AcousticStructureModel, params: Stage4CParameters, *, nra_enabled: bool = True) -> dict[str, np.ndarray]:
    f = np.asarray(list(freqs_Hz), dtype=float)
    Zb = np.asarray(Zb_ohm, dtype=complex)
    unit = solve_coupled_unit_force(model, f, params, nra_enabled=nra_enabled)
    Zm = unit['mechanical_impedance_N_s_m']
    Ztotal = Zb + params.BL_N_A**2/Zm
    I = params.V0_peak_V/Ztotal
    F = params.BL_N_A*I
    omega = 2.0*np.pi*f
    v = unit['velocity_per_N_m_s_per_N']*F
    disp = unit['displacement_per_N'] * F[:, None]
    p_field = unit['pressure_per_N'] * F[:, None]
    # Primary pressure estimate: volume velocity from the ASB structure, consistent with Stage 4B anchors.
    # Boundary-93 HK replacement is kept as Stage-4D; here pressure field is used for near-field diagnostics.
    p_axis = 1j*omega*params.rho0_kg_m3*params.Sd_m2*v/(2.0*math.pi*params.observation_distance_m)
    SPL = 20*np.log10(np.maximum(np.abs(p_axis)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    phase = np.unwrap(np.angle(p_axis))*180.0/math.pi
    coil_power = 0.5*np.real(params.V0_peak_V*np.conj(I))
    # Approximate acoustic power from half-space intensity at 1 m for diagnostics.
    acoustic_power = np.abs(p_axis)**2/(2.0*params.rho0_kg_m3*params.c0_m_s) * 2.0*math.pi*params.observation_distance_m**2
    eff = 100.0*acoustic_power/np.maximum(coil_power, 1e-300)
    return {
        'f_Hz': f,
        'Zb_ohm': Zb,
        'Zm_asb_N_s_m': Zm,
        'Z_total_ohm': Ztotal,
        'I_A_peak': I,
        'F_Lorentz_N_peak': F,
        'v_coil_m_s_peak': v,
        'p_1m_Pa_peak': p_axis,
        'SPL_1m_dB': SPL,
        'phase_deg': phase,
        'coil_power_W': coil_power,
        'acoustic_power_W': acoustic_power,
        'acoustic_efficiency_percent': eff,
        'solid_displacement_m': disp,
        'acoustic_pressure_field_Pa': p_field,
        'unit_force': unit,
    }


def result_to_rows_stage4C(result: Mapping[str, np.ndarray]) -> list[dict]:
    rows=[]
    for i, fi in enumerate(result['f_Hz']):
        Z=result['Z_total_ohm'][i]; Zb=result['Zb_ohm'][i]; Zm=result['Zm_asb_N_s_m'][i]
        rows.append({
            'f_Hz': float(fi),
            'SPL_1m_dB': float(result['SPL_1m_dB'][i]),
            'phase_deg': float(result['phase_deg'][i]),
            'Z_abs_ohm': float(abs(Z)),
            'Z_real_ohm': float(np.real(Z)),
            'Z_imag_ohm': float(np.imag(Z)),
            'Zb_abs_ohm': float(abs(Zb)),
            'Zb_real_ohm': float(np.real(Zb)),
            'Zb_imag_ohm': float(np.imag(Zb)),
            'Zm_abs_N_s_m': float(abs(Zm)),
            'Zm_real_N_s_m': float(np.real(Zm)),
            'Zm_imag_N_s_m': float(np.imag(Zm)),
            'I_abs_A_peak': float(abs(result['I_A_peak'][i])),
            'F_abs_N_peak': float(abs(result['F_Lorentz_N_peak'][i])),
            'v_abs_m_s_peak': float(abs(result['v_coil_m_s_peak'][i])),
            'coil_power_W': float(result['coil_power_W'][i]),
            'acoustic_power_W': float(result['acoustic_power_W'][i]),
            'acoustic_efficiency_percent': float(result['acoustic_efficiency_percent'][i]),
        })
    return rows
