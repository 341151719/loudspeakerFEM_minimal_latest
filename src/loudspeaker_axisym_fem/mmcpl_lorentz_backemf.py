from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Iterable

import numpy as np
from scipy.sparse import bmat, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .axisym_magnetics import TaggedTriMesh, load_tagged_meshio
from .stage4_solid_fem import SolidFEMModel, COIL_DOMAINS, _tri_area_and_grads, _complex_stiffness


@dataclass
class LorentzBackEmfVector:
    """Energy-conjugate Lorentz/back-EMF coupling vector.

    g has units N/A.  For a scalar coil current I, the mechanical force vector is
    F_u = I g.  For a structural velocity vector v = i omega u, the back-EMF is
    V_be = g^T v.  Using the same g on both sides is the reciprocity condition.
    """
    g_full_N_per_A: np.ndarray
    g_free_N_per_A: np.ndarray
    free_dofs: np.ndarray
    Acoil_rz_m2: float
    N0: float
    current_shape_per_m2: float
    axial_BL_N_per_A: float
    radial_resultant_N_per_A: float
    coil_volume_m3: float
    n_coil_triangles: int
    b_source: str
    interpolation: str
    sign_convention: str = "Jphi x B0 = Jphi*(Bz e_r - Br e_z); exp(+i omega t); Vbe = g^T v"

    def summary(self) -> dict:
        d = asdict(self)
        d.pop('g_full_N_per_A', None)
        d.pop('g_free_N_per_A', None)
        d.pop('free_dofs', None)
        d['norm_g_full_N_per_A'] = float(np.linalg.norm(self.g_full_N_per_A))
        d['norm_g_free_N_per_A'] = float(np.linalg.norm(self.g_free_N_per_A))
        d['max_abs_g_entry_N_per_A'] = float(np.max(np.abs(self.g_full_N_per_A))) if len(self.g_full_N_per_A) else 0.0
        return d


def _tri_geometry(points: np.ndarray, tri: np.ndarray):
    p = points[np.asarray(tri, dtype=int)]
    area, grads = _tri_area_and_grads(p)
    centroid = np.mean(p, axis=0)
    return area, centroid, grads


def _load_cell_center_B_from_vtu(vtu_path: str | Path):
    import meshio
    m = meshio.read(str(vtu_path))
    cells = None
    for block in m.cells:
        if block.type == 'triangle':
            cells = np.asarray(block.data, dtype=int)
            break
    if cells is None:
        raise ValueError(f'{vtu_path} contains no triangle cells')
    pts = np.asarray(m.points[:, :2], dtype=float)
    cent = np.mean(pts[cells], axis=1)
    cd = m.cell_data_dict
    # meshio stores VTU cell_data either as dict names or list entries.  The project VTU uses cell_data_dict.
    if 'B_r_T' in cd:
        Br = np.asarray(cd['B_r_T']['triangle'], dtype=float)
        Bz = np.asarray(cd['B_z_T']['triangle'], dtype=float)
        domain = np.asarray(cd.get('domain', {}).get('triangle', np.zeros(len(cells))), dtype=int)
    else:
        # Fallback for direct cell_data lists.
        fields = {k: np.asarray(v[0]) for k, v in m.cell_data.items()}
        Br = np.asarray(fields['B_r_T'], dtype=float)
        Bz = np.asarray(fields['B_z_T'], dtype=float)
        domain = np.asarray(fields.get('domain', np.zeros(len(cells))), dtype=int)
    return cent, Br, Bz, domain


def assemble_lorentz_backemf_vector(
    model: SolidFEMModel,
    magnetostatic_vtu: str | Path,
    *,
    N0: float = 100.0,
    coil_domains: Iterable[int] = COIL_DOMAINS,
    interpolation: str = 'nearest_cell_centroid',
) -> LorentzBackEmfVector:
    """Assemble the DOF-level Lorentz/back-EMF vector g.

    The coil current density is homogenized exactly as the COMSOL tutorial's
    cross-section formula: J_phi = I * N0 / Acoil, where Acoil is the rz-plane
    coil-domain area, not the revolved volume.
    """
    coil_domains = tuple(int(x) for x in coil_domains)
    cent_B, Br_cells, Bz_cells, _ = _load_cell_center_B_from_vtu(magnetostatic_vtu)
    tree = cKDTree(cent_B)

    pts = np.asarray(model.points_rz_m, dtype=float)
    tris = np.asarray(model.triangles, dtype=int)
    doms = np.asarray(model.domains, dtype=int)
    coil_mask = np.isin(doms, coil_domains)
    if not np.any(coil_mask):
        raise RuntimeError('No coil domains found in structural model')

    # First pass: COMSOL homogenized coil uses rz cross-section area.
    Acoil = 0.0
    coil_volume = 0.0
    coil_data = []
    for tri, dom in zip(tris[coil_mask], doms[coil_mask]):
        area, centroid, grads = _tri_geometry(pts, tri)
        if area <= 0:
            continue
        rbar = max(float(centroid[0]), 1e-12)
        Acoil += area
        vol = 2.0 * math.pi * rbar * area
        coil_volume += vol
        coil_data.append((tri, area, centroid, rbar, vol))
    if Acoil <= 0:
        raise RuntimeError('Acoil is zero')
    s = float(N0) / Acoil

    g = np.zeros(model.ndof, dtype=float)
    for tri, area, centroid, rbar, vol in coil_data:
        _, idx = tree.query(centroid, k=1)
        Br = float(Br_cells[int(idx)])
        Bz = float(Bz_cells[int(idx)])
        # J_phi x B0 = J_phi*(Bz e_r - Br e_z)
        fr_per_I = s * Bz
        fz_per_I = s * (-Br)
        nodal = vol / 3.0
        for a in tri:
            ai = int(a)
            g[2*ai + 0] += nodal * fr_per_I
            g[2*ai + 1] += nodal * fz_per_I

    free = np.asarray(model.free_dofs, dtype=int)
    gf = g[free]
    return LorentzBackEmfVector(
        g_full_N_per_A=g,
        g_free_N_per_A=gf,
        free_dofs=free,
        Acoil_rz_m2=float(Acoil),
        N0=float(N0),
        current_shape_per_m2=float(s),
        axial_BL_N_per_A=float(np.sum(g[1::2])),
        radial_resultant_N_per_A=float(np.sum(g[0::2])),
        coil_volume_m3=float(coil_volume),
        n_coil_triangles=int(len(coil_data)),
        b_source=str(magnetostatic_vtu),
        interpolation=interpolation,
    )


def rigid_reciprocity_tests(cpl: LorentzBackEmfVector) -> dict:
    g = cpl.g_full_N_per_A
    vz = np.zeros_like(g)
    vz[1::2] = 1.0
    vr = np.zeros_like(g)
    vr[0::2] = 1.0
    axial_from_backemf = float(np.dot(g, vz))
    radial_from_backemf = float(np.dot(g, vr))
    return {
        'axial_force_sum_N_per_A': float(np.sum(g[1::2])),
        'axial_backemf_for_unit_vz_V_per_m_s': axial_from_backemf,
        'axial_reciprocity_abs_error': abs(float(np.sum(g[1::2])) - axial_from_backemf),
        'radial_force_sum_N_per_A': float(np.sum(g[0::2])),
        'radial_backemf_for_unit_vr_V_per_m_s': radial_from_backemf,
        'radial_reciprocity_abs_error': abs(float(np.sum(g[0::2])) - radial_from_backemf),
    }


def power_reciprocity_test(cpl: LorentzBackEmfVector, *, omega: float = 2.0*math.pi*50.0, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    g = cpl.g_free_N_per_A.astype(float)
    u = rng.normal(size=len(g)) + 1j*rng.normal(size=len(g))
    I = 0.73 - 0.21j
    v = 1j * omega * u
    F = I * g
    Vbe = np.dot(g, v)
    P_mech = 0.5 * np.real(np.dot(F, np.conj(v)))
    P_back = 0.5 * np.real(Vbe * np.conj(I))
    denom = max(abs(P_mech), abs(P_back), 1e-300)
    return {
        'omega_rad_s': float(omega),
        'I_test_A': [float(I.real), float(I.imag)],
        'P_mechanical_W': float(P_mech),
        'P_back_emf_W': float(P_back),
        'abs_difference_W': float(abs(P_mech - P_back)),
        'relative_difference': float(abs(P_mech - P_back) / denom),
    }


def assemble_minimal_mmcpl_block(model: SolidFEMModel, cpl: LorentzBackEmfVector, Zb: complex, freq_Hz: float):
    """Assemble [I, u_free] electromechanical block using Zb and the same g on both sides.

    Unknown ordering is x = [I, u_free].  The equations are
      Zb I + i omega g^T u = V
      -g I + H_u u = 0
    """
    omega = 2.0 * math.pi * float(freq_Hz)
    gf = np.asarray(cpl.g_free_N_per_A, dtype=complex)
    free = cpl.free_dofs
    H = (_complex_stiffness(model, omega)[free][:, free].astype(complex)
         - (omega*omega) * model.M[free][:, free].astype(complex)).tocsr()
    A00 = csr_matrix([[complex(Zb)]])
    A01 = csr_matrix((1j*omega*gf.reshape(1, -1)))
    A10 = csr_matrix((-gf.reshape(-1, 1)))
    A = bmat([[A00, A01], [A10, H]], format='csc')
    return A


def solve_fixed_structure_regression(freqs_Hz: Iterable[float], Zb: Iterable[complex], V0: float = 3.55) -> dict[str, np.ndarray]:
    freqs = np.asarray(list(freqs_Hz), dtype=float)
    Zb = np.asarray(list(Zb), dtype=complex)
    I = V0 / Zb
    Zreg = V0 / I
    return {
        'f_Hz': freqs,
        'Z_input_ohm': Zb,
        'I_fixed_A_peak': I,
        'Z_regressed_ohm': Zreg,
        'abs_error_ohm': np.abs(Zreg - Zb),
        'rel_error': np.abs(Zreg - Zb) / np.maximum(np.abs(Zb), 1e-300),
    }


def solve_mmcpl_block_for_frequency(model: SolidFEMModel, cpl: LorentzBackEmfVector, Zb: complex, freq_Hz: float, V0: float = 3.55):
    A = assemble_minimal_mmcpl_block(model, cpl, Zb, freq_Hz)
    rhs = np.zeros(A.shape[0], dtype=complex)
    rhs[0] = V0
    sol = spsolve(A, rhs)
    I = sol[0]
    u = sol[1:]
    Z = V0 / I
    return I, u, Z
