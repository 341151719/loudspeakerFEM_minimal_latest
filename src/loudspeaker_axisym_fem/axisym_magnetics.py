from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * math.pi


def interp_bh(H: np.ndarray | float, table: Iterable[Tuple[float, float]]) -> np.ndarray:
    tab = np.asarray(list(table), dtype=float)
    h = tab[:, 0]
    b = tab[:, 1]
    return np.interp(H, h, b)


def interp_h_from_b(B: np.ndarray | float, table: Iterable[Tuple[float, float]]) -> np.ndarray:
    """Inverse B-H interpolation H(B) using COMSOL's tabulated curve.

    The original Stage-2 Picard update estimated H as B/(mu0*mu_old) and then
    evaluated B(H)/H.  For a first-principles fixed point, the local secant
    permeability should instead satisfy H = BH_inv(|B|).  This helper is used
    by the RAW-BL ABCDE audit and by ``nonlinear_update_mode='B_inverse'``.
    """
    tab = np.asarray(list(table), dtype=float)
    h = tab[:, 0]
    b = tab[:, 1]
    return np.interp(np.asarray(B, dtype=float), b, h)


def effective_mu_r_from_B(B: np.ndarray | float, table: Iterable[Tuple[float, float]], *, floor: float = 1.0, cap: float = 4000.0) -> np.ndarray:
    Barr = np.asarray(B, dtype=float)
    H = np.maximum(interp_h_from_b(np.maximum(Barr, 0.0), table), 1e-30)
    mu = Barr / (MU0 * H)
    return np.clip(np.maximum(mu, floor), floor, cap)


def effective_mu_r(H: np.ndarray | float, table: Iterable[Tuple[float, float]], *, floor: float = 1.0) -> np.ndarray:
    H_arr = np.asarray(H, dtype=float)
    B = interp_bh(H_arr, table)
    out = np.empty_like(H_arr, dtype=float)
    mask = np.abs(H_arr) > 1e-30
    out[mask] = B[mask] / (MU0 * np.maximum(np.abs(H_arr[mask]), 1e-30))
    out[~mask] = floor
    return np.maximum(out, floor)


def differential_mu_r(H: np.ndarray | float, table: Iterable[Tuple[float, float]]) -> np.ndarray:
    tab = np.asarray(list(table), dtype=float)
    h = tab[:, 0]
    b = tab[:, 1]
    slopes = np.gradient(b, h, edge_order=1)
    slope = np.interp(np.asarray(H, dtype=float), h, slopes)
    return slope / MU0


def skin_depth_m(f_Hz: float, sigma_S_m: float, mu_r: float) -> float:
    omega = 2.0 * math.pi * f_Hz
    return math.sqrt(2.0 / (omega * MU0 * mu_r * sigma_S_m))


def blocked_inductance_from_impedance(Zb: np.ndarray, freqs_Hz: np.ndarray) -> np.ndarray:
    omega = 2.0 * math.pi * np.asarray(freqs_Hz, dtype=float)
    return np.imag(Zb) / np.maximum(omega, 1e-300)


@dataclass
class TaggedTriMesh:
    points_rz_m: np.ndarray  # shape (n, 2)
    triangles: np.ndarray    # shape (nt, 3)
    tri_domains: np.ndarray  # COMSOL domain id per triangle
    line_cells: np.ndarray
    line_tags: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.points_rz_m.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def boundary_nodes(self, boundary_ids: Optional[Iterable[int]] = None) -> np.ndarray:
        if self.line_cells.size == 0:
            return np.array([], dtype=int)
        if boundary_ids is None:
            mask = np.ones(len(self.line_tags), dtype=bool)
        else:
            ids = set(int(x) for x in boundary_ids)
            mask = np.array([int(t) in ids for t in self.line_tags], dtype=bool)
        if not np.any(mask):
            return np.array([], dtype=int)
        return np.unique(self.line_cells[mask].ravel())


def load_tagged_meshio(path: str | Path, *, scale: float = 1.0) -> TaggedTriMesh:
    """Load a gmsh/meshio mesh preserving COMSOL physical surface/curve ids.

    The Stage-1 geometry exporter writes coordinates in meters.  ``scale`` is
    kept for defensive use when loading other versions.
    """
    import meshio

    m = meshio.read(str(path))
    points = np.asarray(m.points[:, :2], dtype=float) * float(scale)
    tri_blocks: List[np.ndarray] = []
    line_blocks: List[np.ndarray] = []
    tri_tags: List[np.ndarray] = []
    line_tags: List[np.ndarray] = []
    phys = m.cell_data_dict.get("gmsh:physical", {})
    tri_seen = 0
    quad_seen = 0
    line_seen = 0
    for ib, block in enumerate(m.cells):
        if block.type == "triangle":
            arr = np.asarray(block.data, dtype=int)
            tri_blocks.append(arr)
            tags_all = phys.get("triangle")
            if tags_all is None:
                tri_tags.append(np.zeros(len(arr), dtype=int))
            else:
                tri_tags.append(np.asarray(tags_all[tri_seen:tri_seen + len(arr)], dtype=int))
            tri_seen += len(arr)
        elif block.type == "quad":
            # Gmsh BoundaryLayer/Transfinite meshing may create quads.
            # The current magnetic FEM assembly is triangular P1, so split each
            # quad into two triangles while preserving the physical surface tag.
            q = np.asarray(block.data, dtype=int)
            tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            tri_blocks.append(tris)
            tags_all = phys.get("quad")
            if tags_all is None:
                tri_tags.append(np.zeros(len(tris), dtype=int))
            else:
                tags = np.asarray(
                    tags_all[quad_seen:quad_seen + len(q)], dtype=int
                )
                # ``tris`` is stacked diagonal-wise: first every [0,1,2]
                # triangle, then every [0,2,3] triangle.  Tile the domain-tag
                # vector in that same order.  ``repeat`` would silently assign
                # the wrong physical domains whenever adjacent quads differ.
                tri_tags.append(np.tile(tags, 2))
            quad_seen += len(q)
        elif block.type == "line":
            arr = np.asarray(block.data, dtype=int)
            line_blocks.append(arr)
            tags_all = phys.get("line")
            if tags_all is None:
                line_tags.append(np.zeros(len(arr), dtype=int))
            else:
                line_tags.append(np.asarray(tags_all[line_seen:line_seen + len(arr)], dtype=int))
            line_seen += len(arr)
    if not tri_blocks:
        raise ValueError(f"mesh {path} contains no triangle cells")
    triangles = np.vstack(tri_blocks)
    domains = np.concatenate(tri_tags)
    line_cells = np.vstack(line_blocks) if line_blocks else np.empty((0, 2), dtype=int)
    line_physical = np.concatenate(line_tags) if line_tags else np.empty((0,), dtype=int)
    return TaggedTriMesh(points, triangles, domains, line_cells, line_physical)


@dataclass
class MagnetostaticResult:
    mesh: TaggedTriMesh
    A_phi: np.ndarray
    B_r: np.ndarray
    B_z: np.ndarray
    B_norm: np.ndarray
    H_norm: np.ndarray
    mu_r_elem: np.ndarray
    rhs_scale: float
    iterations: int
    residual_history: List[float]
    bl_raw_N_A: float
    bl_calibrated_N_A: float
    calibration_factor: float
    remanence_T: float

    def summary(self) -> dict:
        return {
            "n_nodes": self.mesh.n_nodes,
            "n_triangles": self.mesh.n_triangles,
            "iterations": self.iterations,
            "residual_history": self.residual_history,
            "remanence_T": self.remanence_T,
            "rhs_scale": self.rhs_scale,
            "BL_raw_N_per_A": self.bl_raw_N_A,
            "BL_calibrated_N_per_A": self.bl_calibrated_N_A,
            "calibration_factor_to_target": self.calibration_factor,
            "B_norm_max_T": float(np.nanmax(self.B_norm)),
            "H_norm_max_A_m": float(np.nanmax(self.H_norm)),
            "mu_r_elem_max": float(np.nanmax(self.mu_r_elem)),
            "mu_r_elem_min": float(np.nanmin(self.mu_r_elem)),
        }


class MagneticsNotYetAssembled(RuntimeError):
    pass


def _tri_geometry(points: np.ndarray, tris: np.ndarray):
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]
    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * np.abs(det)
    # Gradients of linear shape functions.  Sign must use signed det.
    dNdx = np.empty((len(tris), 3), dtype=float)
    dNdy = np.empty((len(tris), 3), dtype=float)
    dNdx[:, 0] = (y1 - y2) / det
    dNdx[:, 1] = (y2 - y0) / det
    dNdx[:, 2] = (y0 - y1) / det
    dNdy[:, 0] = (x2 - x1) / det
    dNdy[:, 1] = (x0 - x2) / det
    dNdy[:, 2] = (x1 - x0) / det
    centroid = (p0 + p1 + p2) / 3.0
    return area, centroid, dNdx, dNdy


def _assemble_linear_system(
    mesh: TaggedTriMesh,
    mu_r_elem: np.ndarray,
    *,
    magnet_domains: Iterable[int],
    remanence_T: float,
    dirichlet_nodes: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray]:
    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, dNdr, dNdz = _tri_geometry(pts, tris)
    r = np.maximum(centroid[:, 0], 1e-9)
    nu = 1.0 / (MU0 * np.maximum(mu_r_elem, 1.0))
    magnet_set = set(int(x) for x in magnet_domains)
    is_magnet = np.array([int(d) in magnet_set for d in mesh.tri_domains], dtype=bool)
    n = mesh.n_nodes
    rows = []
    cols = []
    data = []
    rhs = np.zeros(n, dtype=float)
    Ncent = 1.0 / 3.0
    two_pi = 2.0 * math.pi
    for e in range(mesh.n_triangles):
        if area[e] <= 0:
            continue
        nodes = tris[e]
        weight = two_pi * r[e] * area[e] * nu[e]
        gz = dNdz[e]
        gr_plus = dNdr[e] + Ncent / r[e]
        ke = weight * (np.outer(gz, gz) + np.outer(gr_plus, gr_plus))
        # Permanent magnet remanence in +z: weak RHS ∫2πr ν Br δBz.
        if is_magnet[e] and remanence_T != 0.0:
            fe = weight * remanence_T * gr_plus
            rhs[nodes] += fe
        for a in range(3):
            for b in range(3):
                rows.append(nodes[a])
                cols.append(nodes[b])
                data.append(ke[a, b])
    K = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # Essential boundary condition A_phi = 0 on exterior and axis.
    fixed = np.unique(np.asarray(dirichlet_nodes, dtype=int))
    if fixed.size:
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed] = False
        free = np.nonzero(free_mask)[0]
        Kff = K[free][:, free].tocsr()
        bf = rhs[free]
        return Kff, bf, free
    return K, rhs, np.arange(n)


def _element_fields(mesh: TaggedTriMesh, A: np.ndarray, mu_r_elem: np.ndarray):
    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, dNdr, dNdz = _tri_geometry(pts, tris)
    r = np.maximum(centroid[:, 0], 1e-9)
    Ae = A[tris]
    dAdr = np.einsum('ei,ei->e', Ae, dNdr)
    dAdz = np.einsum('ei,ei->e', Ae, dNdz)
    Acent = np.mean(Ae, axis=1)
    B_r = -dAdz
    B_z = dAdr + Acent / r
    B_norm = np.sqrt(B_r ** 2 + B_z ** 2)
    H_norm = B_norm / (MU0 * np.maximum(mu_r_elem, 1.0))
    return B_r, B_z, B_norm, H_norm


def _default_dirichlet_nodes(mesh: TaggedTriMesh, exterior_boundary_ids: Iterable[int]) -> np.ndarray:
    r = mesh.points_rz_m[:, 0]
    axis = np.nonzero(np.abs(r) < 1e-10)[0]
    exterior = mesh.boundary_nodes(exterior_boundary_ids)
    return np.unique(np.concatenate([axis, exterior]))


def compute_bl_from_elements(
    mesh: TaggedTriMesh,
    B_r: np.ndarray,
    *,
    coil_domains: Iterable[int],
    N0: int,
) -> float:
    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, _, _ = _tri_geometry(pts, tris)
    coil_set = set(int(x) for x in coil_domains)
    mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    if not np.any(mask):
        return float('nan')
    area_total = float(np.sum(area[mask]))
    val = np.sum((-2.0 * math.pi * float(N0) * centroid[mask, 0] * B_r[mask]) * area[mask]) / area_total
    return float(val)


def solve_axisymmetric_magnetostatics(
    mesh: TaggedTriMesh,
    *,
    soft_iron_domains: Iterable[int] = (6, 23),
    magnet_domains: Iterable[int] = (24,),
    coil_domains: Iterable[int] = (17, 18, 19),
    N0: int = 100,
    remanence_T: float = 0.4,
    target_BL_N_A: float = 10.48,
    bh_table: Iterable[Tuple[float, float]],
    exterior_boundary_ids: Optional[Iterable[int]] = None,
    max_iter: int = 8,
    tol: float = 5e-3,
    relaxation: float = 0.55,
    mu_r_initial_soft: float = 700.0,
    mu_r_air: float = 1.0,
    calibrate_to_BL: bool = False,
    nonlinear_update_mode: str = "H_forward",
    remanence_rhs_sign: float = 1.0,
) -> MagnetostaticResult:
    """Solve a scalar A_phi axisymmetric magnetostatic model.

    This is a real FEM assembly, but still a Stage-2 reproduction: nonlinear
    soft-iron is handled by Picard updates using the COMSOL B-H table.  The
    calibrated result is reported separately so raw deviations from COMSOL's
    BL=10.48 N/A remain visible.
    """
    tri_domains = mesh.tri_domains.astype(int)
    soft_set = set(int(x) for x in soft_iron_domains)
    mu_r = np.full(mesh.n_triangles, float(mu_r_air), dtype=float)
    soft = np.array([int(d) in soft_set for d in tri_domains], dtype=bool)
    mu_r[soft] = float(mu_r_initial_soft)
    # Ferrite relative recoil permeability is close to 1 in the COMSOL model.
    if exterior_boundary_ids is None:
        # From Stage-1 inventory: boundary edges with one adjacent domain = 0.
        exterior_boundary_ids = (1,2,3,4,5,83,84,85,86,87,88,89,94)
    fixed = _default_dirichlet_nodes(mesh, exterior_boundary_ids)
    A = np.zeros(mesh.n_nodes, dtype=float)
    residuals: List[float] = []
    free = None
    for it in range(1, max_iter + 1):
        Kff, bf, free = _assemble_linear_system(mesh, mu_r, magnet_domains=magnet_domains, remanence_T=remanence_T * float(remanence_rhs_sign), dirichlet_nodes=fixed)
        Af = spsolve(Kff, bf)
        A_new = np.zeros(mesh.n_nodes, dtype=float)
        A_new[free] = Af
        denom = max(float(np.linalg.norm(A_new)), 1e-30)
        rel = float(np.linalg.norm(A_new - A) / denom)
        residuals.append(rel)
        A = A_new
        B_r, B_z, B_norm, H_norm = _element_fields(mesh, A, mu_r)
        if np.any(soft):
            mu_new = mu_r.copy()
            if nonlinear_update_mode == "H_forward":
                # Legacy Picard update: use H estimated from the previous secant mu.
                mu_eff = effective_mu_r(np.maximum(H_norm[soft], 1.0), bh_table, floor=1.0)
                mu_eff = np.clip(mu_eff, 1.0, 4000.0)
            elif nonlinear_update_mode == "B_inverse":
                # First-principles secant update: enforce H = BH_inv(|B|).
                mu_eff = effective_mu_r_from_B(np.maximum(B_norm[soft], 0.0), bh_table, floor=1.0, cap=4000.0)
            else:
                raise ValueError("nonlinear_update_mode must be 'H_forward' or 'B_inverse'")
            mu_new[soft] = (1.0 - relaxation) * mu_r[soft] + relaxation * mu_eff
            mu_r = mu_new
        if rel < tol and it >= 2:
            break
    B_r, B_z, B_norm, H_norm = _element_fields(mesh, A, mu_r)
    bl_raw = compute_bl_from_elements(mesh, B_r, coil_domains=coil_domains, N0=N0)
    # The PDE is almost linear in remanence; report a calibrated field for
    # comparison to COMSOL's hard BL anchor without hiding raw error.
    if calibrate_to_BL and np.isfinite(bl_raw) and abs(bl_raw) > 1e-12:
        factor = float(target_BL_N_A / bl_raw)
    else:
        factor = 1.0
    B_r_cal = B_r * factor
    B_z_cal = B_z * factor
    B_norm_cal = B_norm * abs(factor)
    H_norm_cal = H_norm * abs(factor)
    bl_cal = bl_raw * factor
    return MagnetostaticResult(
        mesh=mesh,
        A_phi=A * factor,
        B_r=B_r_cal,
        B_z=B_z_cal,
        B_norm=B_norm_cal,
        H_norm=H_norm_cal,
        mu_r_elem=mu_r,
        rhs_scale=factor,
        iterations=it,
        residual_history=residuals,
        bl_raw_N_A=bl_raw,
        bl_calibrated_N_A=bl_cal,
        calibration_factor=factor,
        remanence_T=remanence_T * factor,
    )


def write_element_vtu(path: str | Path, result: MagnetostaticResult) -> None:
    import meshio
    pts = np.column_stack([result.mesh.points_rz_m, np.zeros(result.mesh.n_nodes)])
    cells = [("triangle", result.mesh.triangles)]
    cell_data = {
        "domain": [result.mesh.tri_domains.astype(int)],
        "B_r_T": [result.B_r],
        "B_z_T": [result.B_z],
        "B_norm_T": [result.B_norm],
        "H_norm_A_m": [result.H_norm],
        "mu_r": [result.mu_r_elem],
    }
    point_data = {"A_phi_Wb_per_m": result.A_phi}
    meshio.write(str(path), meshio.Mesh(pts, cells, point_data=point_data, cell_data=cell_data))


def plot_magnetic_fields(path_prefix: str | Path, result: MagnetostaticResult, *, title_suffix: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    pts = result.mesh.points_rz_m
    tri = result.mesh.triangles
    triang = mtri.Triangulation(pts[:, 0] * 1e3, pts[:, 1] * 1e3, tri)
    fields = [
        ("H_norm_A_m", result.H_norm, "Magnetic field norm H [A/m]"),
        ("B_norm_T", result.B_norm, "Magnetic flux density norm B [T]"),
        ("mu_r", result.mu_r_elem, "Effective relative permeability [-]"),
        ("B_r_T", result.B_r, "Radial flux density Br [T]"),
    ]
    prefix = Path(path_prefix)
    for name, val, label in fields:
        fig, ax = plt.subplots(figsize=(7, 8))
        tc = ax.tripcolor(triang, facecolors=val, shading='flat')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('r [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_title(f'{label}{title_suffix}')
        fig.colorbar(tc, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + f'_{name}.png'), dpi=220)
        plt.close(fig)


def assemble_axisymmetric_magnetostatics(*args, **kwargs):
    """Backward compatible alias for the implemented Stage-2 solver."""
    return solve_axisymmetric_magnetostatics(*args, **kwargs)

@dataclass
class BlockedImpedanceResult:
    mesh: TaggedTriMesh
    frequencies_Hz: np.ndarray
    Zb_ohm: np.ndarray
    Lb_H: np.ndarray
    Rdc_ohm: float
    coil_area_m2: float
    coil_flux_linkage_Wb_per_A: np.ndarray
    A_phi_by_frequency: Dict[float, np.ndarray]
    Jphi_eddy_by_frequency: Dict[float, np.ndarray]
    mu_r_linearized_elem: np.ndarray
    sigma_elem: np.ndarray
    Zb_raw_ohm: Optional[np.ndarray] = None
    Lb_raw_H: Optional[np.ndarray] = None
    core_inductance_scale: float = 1.0
    leakage_inductance_H: float = 0.0
    calibration_note: str = "none"
    terminal_mode: str = "current_driven"
    voltage_V: float = 1.0
    coil_current_A: Optional[np.ndarray] = None

    def summary(self) -> dict:
        return {
            "n_nodes": self.mesh.n_nodes,
            "n_triangles": self.mesh.n_triangles,
            "frequencies_Hz": [float(x) for x in self.frequencies_Hz],
            "Rdc_ohm": float(self.Rdc_ohm),
            "coil_area_m2": float(self.coil_area_m2),
            "Zb_abs_ohm_first_last": [float(abs(self.Zb_ohm[0])), float(abs(self.Zb_ohm[-1]))] if len(self.Zb_ohm) else [],
            "Lb_mH_first_last": [float(self.Lb_H[0] * 1e3), float(self.Lb_H[-1] * 1e3)] if len(self.Lb_H) else [],
            "Lb_mH_min_max": [float(np.nanmin(self.Lb_H) * 1e3), float(np.nanmax(self.Lb_H) * 1e3)] if len(self.Lb_H) else [],
            "core_inductance_scale": float(self.core_inductance_scale),
            "leakage_inductance_mH": float(self.leakage_inductance_H * 1e3),
            "calibration_note": self.calibration_note,
            "terminal_mode": self.terminal_mode,
            "voltage_V": float(self.voltage_V),
            "coil_current_abs_A_first_last": [float(abs(self.coil_current_A[0])), float(abs(self.coil_current_A[-1]))] if self.coil_current_A is not None and len(self.coil_current_A) else [],
            "raw_Lb_mH_first_last": [float(self.Lb_raw_H[0] * 1e3), float(self.Lb_raw_H[-1] * 1e3)] if self.Lb_raw_H is not None and len(self.Lb_raw_H) else [],
        }


def _coil_area(mesh: TaggedTriMesh, coil_domains: Iterable[int]) -> float:
    area, _, _, _ = _tri_geometry(mesh.points_rz_m, mesh.triangles)
    coil_set = set(int(x) for x in coil_domains)
    mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    return float(np.sum(area[mask]))


def _assemble_frequency_matrices(
    mesh: TaggedTriMesh,
    mu_r_elem: np.ndarray,
    sigma_elem: np.ndarray,
    *,
    coil_domains: Iterable[int],
    N0: int,
    unit_current_A: float,
    dirichlet_nodes: np.ndarray,
    reluctivity_tensor_elem: np.ndarray | None = None,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, float]:
    """Assemble K, M_sigma and unit-current coil source for A_phi perturbation.

    The equation is the magnetoquasistatic frequency-domain form

        curl(nu curl A_phi) + i omega sigma A_phi = J_phi,src

    in the axisymmetric weak form.  The coil is driven by an impressed
    homogenized multi-turn current density J_phi = N0*I/A_coil.  This is not
    yet COMSOL's exact voltage-constrained Coil feature, but it is the correct
    current-driven perturbation building block for blocked inductance and eddy
    current skin-effect validation.
    """
    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, dNdr, dNdz = _tri_geometry(pts, tris)
    r = np.maximum(centroid[:, 0], 1e-9)
    nu = 1.0 / (MU0 * np.maximum(mu_r_elem, 1.0))
    if reluctivity_tensor_elem is not None:
        nu_tensor = np.asarray(reluctivity_tensor_elem, dtype=float)
        if nu_tensor.shape != (mesh.n_triangles, 2, 2):
            raise ValueError(
                "reluctivity_tensor_elem must have shape "
                f"({mesh.n_triangles}, 2, 2), got {nu_tensor.shape}"
            )
    else:
        nu_tensor = None
    n = mesh.n_nodes
    rows = []
    cols = []
    kdata = []
    mdata = []
    rhs = np.zeros(n, dtype=float)
    coil_set = set(int(x) for x in coil_domains)
    coil_mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    Acoil = float(np.sum(area[coil_mask]))
    if Acoil <= 0:
        raise ValueError("coil area is zero; check coil domain ids")
    Jsrc = float(N0) * float(unit_current_A) / Acoil
    two_pi = 2.0 * math.pi
    # linear triangle consistent mass matrix over a triangle
    mass_template = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
    Ncent = 1.0 / 3.0
    for e in range(mesh.n_triangles):
        if area[e] <= 0:
            continue
        nodes = tris[e]
        gz = dNdz[e]
        gr_plus = dNdr[e] + Ncent / r[e]
        weightK = two_pi * r[e] * area[e]
        if nu_tensor is None:
            ke = weightK * nu[e] * (np.outer(gz, gz) + np.outer(gr_plus, gr_plus))
        else:
            # For axisymmetric A_phi, [B_r, B_z] =
            # [-d(A_phi)/dz, d(A_phi)/dr + A_phi/r].
            q_r = -gz
            q_z = gr_plus
            nt = nu_tensor[e]
            ke = weightK * (
                nt[0, 0] * np.outer(q_r, q_r)
                + nt[0, 1] * np.outer(q_r, q_z)
                + nt[1, 0] * np.outer(q_z, q_r)
                + nt[1, 1] * np.outer(q_z, q_z)
            )
        me = two_pi * r[e] * area[e] * sigma_elem[e] * mass_template
        if coil_mask[e]:
            rhs[nodes] += two_pi * r[e] * area[e] * Jsrc * Ncent
        for a in range(3):
            for b in range(3):
                rows.append(nodes[a]); cols.append(nodes[b]); kdata.append(ke[a, b]); mdata.append(me[a, b])
    K = coo_matrix((kdata, (rows, cols)), shape=(n, n)).tocsr()
    M = coo_matrix((mdata, (rows, cols)), shape=(n, n)).tocsr()
    fixed = np.unique(np.asarray(dirichlet_nodes, dtype=int))
    free_mask = np.ones(n, dtype=bool)
    free_mask[fixed] = False
    free = np.nonzero(free_mask)[0]
    return K[free][:, free].tocsr(), M[free][:, free].tocsr(), rhs[free], free, Acoil


def _flux_linkage_from_A(mesh: TaggedTriMesh, A: np.ndarray, *, coil_domains: Iterable[int], N0: int, coil_area_m2: float) -> complex:
    tris = mesh.triangles
    area, centroid, _, _ = _tri_geometry(mesh.points_rz_m, tris)
    coil_set = set(int(x) for x in coil_domains)
    mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    if not np.any(mask):
        return complex('nan')
    Acent = np.mean(A[tris], axis=1)
    val = np.sum(2.0 * math.pi * centroid[mask, 0] * Acent[mask] * area[mask])
    return complex(float(N0) * val / max(coil_area_m2, 1e-300))


def _elem_centroid_values(mesh: TaggedTriMesh, nodal: np.ndarray) -> np.ndarray:
    return np.mean(nodal[mesh.triangles], axis=1)


def linearized_mu_from_static(
    static_result: MagnetostaticResult,
    *,
    soft_iron_domains: Iterable[int],
    bh_table: Iterable[Tuple[float, float]],
    mode: str = "differential",
    floor: float = 1.0,
    cap: float = 4000.0,
) -> np.ndarray:
    """Return element mu_r for frequency perturbation around the static field."""
    mu = np.ones(static_result.mesh.n_triangles, dtype=float)
    soft_set = set(int(x) for x in soft_iron_domains)
    soft = np.array([int(d) in soft_set for d in static_result.mesh.tri_domains], dtype=bool)
    H = np.maximum(static_result.H_norm[soft], 1.0)
    if mode == "differential":
        m = differential_mu_r(H, bh_table)
    elif mode == "effective":
        m = effective_mu_r(H, bh_table, floor=floor)
    elif mode == "stage2":
        m = static_result.mu_r_elem[soft]
    else:
        raise ValueError(f"unknown linearized mu mode {mode!r}")
    mu[soft] = np.clip(np.maximum(np.real(m), floor), floor, cap)
    return mu


def tangent_reluctivity_tensor_from_static(
    static_result: MagnetostaticResult,
    *,
    soft_iron_domains: Iterable[int],
    bh_table: Iterable[Tuple[float, float]],
    floor: float = 1.0,
    cap: float = 4000.0,
    anisotropy_factor: float = 1.0,
) -> np.ndarray:
    """Incremental reluctivity tensor about the DC-biased nonlinear solution.

    An isotropic nonlinear B-H law becomes anisotropic after linearization:
    the perturbation parallel to the static B field sees differential
    permeability dB/dH, while the perpendicular perturbation sees secant
    permeability B/H.  Treating both directions as differential permeability
    is only valid when the AC and DC flux directions coincide everywhere.
    """
    n_elem = static_result.mesh.n_triangles
    nu0 = 1.0 / MU0
    tensor = np.zeros((n_elem, 2, 2), dtype=float)
    tensor[:, 0, 0] = nu0
    tensor[:, 1, 1] = nu0

    soft_set = set(int(x) for x in soft_iron_domains)
    soft = np.array(
        [int(d) in soft_set for d in static_result.mesh.tri_domains],
        dtype=bool,
    )
    if not np.any(soft):
        return tensor

    H = np.maximum(np.asarray(static_result.H_norm[soft], dtype=float), 1.0)
    mu_parallel = np.clip(differential_mu_r(H, bh_table), floor, cap)
    mu_perpendicular = np.clip(
        effective_mu_r(H, bh_table, floor=floor), floor, cap
    )
    nu_parallel = 1.0 / (MU0 * mu_parallel)
    nu_perpendicular = 1.0 / (MU0 * mu_perpendicular)

    br = np.asarray(static_result.B_r[soft], dtype=float)
    bz = np.asarray(static_result.B_z[soft], dtype=float)
    bn = np.hypot(br, bz)
    er = np.divide(br, bn, out=np.zeros_like(br), where=bn > 1e-15)
    ez = np.divide(bz, bn, out=np.ones_like(bz), where=bn > 1e-15)
    alpha = float(anisotropy_factor)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("anisotropy_factor must be between 0 and 1")

    # alpha=0 reproduces the legacy isotropic differential-permeability
    # operator; alpha=1 is the full tensor linearization.
    nu_perpendicular = (
        (1.0 - alpha) * nu_parallel + alpha * nu_perpendicular
    )
    delta = alpha * (nu_parallel - 1.0 / (MU0 * mu_perpendicular))

    idx = np.nonzero(soft)[0]
    tensor[idx, 0, 0] = nu_perpendicular + delta * er * er
    tensor[idx, 0, 1] = delta * er * ez
    tensor[idx, 1, 0] = delta * er * ez
    tensor[idx, 1, 1] = nu_perpendicular + delta * ez * ez
    return tensor




def _flux_linkage_vector(mesh: TaggedTriMesh, *, coil_domains: Iterable[int], N0: int, coil_area_m2: float) -> np.ndarray:
    """Vector c such that flux linkage lambda = c^T A.

    Uses the same centroid/mass-lumped quadrature as _flux_linkage_from_A so
    current-driven and voltage-driven formulations are algebraically identical
    when using the same mesh and source normalization.
    """
    tris = mesh.triangles
    area, centroid, _, _ = _tri_geometry(mesh.points_rz_m, tris)
    coil_set = set(int(x) for x in coil_domains)
    mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    c = np.zeros(mesh.n_nodes, dtype=float)
    if not np.any(mask):
        return c
    for e in np.nonzero(mask)[0]:
        coeff = float(N0) * 2.0 * math.pi * centroid[e, 0] * area[e] / (3.0 * max(coil_area_m2, 1e-300))
        c[tris[e]] += coeff
    return c


def solve_voltage_constrained_blocked_coil_impedance(
    static_result: MagnetostaticResult,
    frequencies_Hz: Iterable[float],
    *,
    bh_table: Iterable[Tuple[float, float]],
    soft_iron_domains: Iterable[int] = (6, 23),
    conducting_domains: Iterable[int] = (6, 23),
    coil_domains: Iterable[int] = (17, 18, 19),
    N0: int = 100,
    Rdc_ohm: float = 5.6,
    sigma_soft_iron_S_m: float = 1.12e7,
    linearized_mu_mode: str = "differential",
    voltage_V: float = 1.0,
    exterior_boundary_ids: Optional[Iterable[int]] = None,
    store_field_frequencies: Iterable[float] = (50.0, 900.0),
    solve_mode: str = "schur",
) -> BlockedImpedanceResult:
    """Solve the homogenized coil as a voltage-constrained terminal.

    The exact coupled equations are

        (K + iωM) A - b I = 0
        iω c^T A + Rdc I = V

    ``solve_mode='schur'`` uses the exact Schur complement of this global
    terminal system: first solve y=(K+iωM)^-1 b, then
    I=V/(Rdc+iω c^T y), A=y I.  This is algebraically identical to the
    monolithic global-unknown matrix but avoids forming an ill-conditioned
    64k+1 saddle matrix for every frequency.  ``solve_mode='monolithic'`` is
    retained for small debug meshes.
    """
    mesh = static_result.mesh
    freqs = np.asarray(list(frequencies_Hz), dtype=float)
    if exterior_boundary_ids is None:
        exterior_boundary_ids = (1,2,3,4,5,83,84,85,86,87,88,89,94)
    fixed = _default_dirichlet_nodes(mesh, exterior_boundary_ids)
    mu_lin = linearized_mu_from_static(static_result, soft_iron_domains=soft_iron_domains, bh_table=bh_table, mode=linearized_mu_mode)
    sigma = np.zeros(mesh.n_triangles, dtype=float)
    cond_set = set(int(x) for x in conducting_domains)
    cond = np.array([int(d) in cond_set for d in mesh.tri_domains], dtype=bool)
    sigma[cond] = float(sigma_soft_iron_S_m)
    K, M, b_free, free, Acoil = _assemble_frequency_matrices(
        mesh, mu_lin, sigma,
        coil_domains=coil_domains, N0=N0, unit_current_A=1.0, dirichlet_nodes=fixed,
    )
    c_full = _flux_linkage_vector(mesh, coil_domains=coil_domains, N0=N0, coil_area_m2=Acoil)
    c_free = c_full[free].astype(complex)
    if solve_mode not in ("schur", "monolithic"):
        raise ValueError("solve_mode must be 'schur' or 'monolithic'")
    Zs=[]; Ls=[]; fluxes=[]; currents=[]
    A_store: Dict[float, np.ndarray] = {}
    J_store: Dict[float, np.ndarray] = {}
    store_targets=[float(x) for x in store_field_frequencies]
    nfree=len(free)
    b_complex=b_free.astype(complex)
    if solve_mode == "monolithic":
        from scipy.sparse import bmat, csc_matrix
        b_col = csc_matrix((-b_complex).reshape(-1,1))
        rhs_base = np.zeros(nfree+1, dtype=complex)
        rhs_base[-1] = complex(voltage_V)
    for f in freqs:
        omega=2.0*math.pi*float(f)
        Aop=(K.astype(complex)+1j*omega*M.astype(complex)).tocsc()
        if solve_mode == "schur":
            y=spsolve(Aop, b_complex)  # field per 1 A terminal current
            lam_unit=complex(c_free @ y)
            Z=complex(Rdc_ohm)+1j*omega*lam_unit
            I=complex(voltage_V)/Z if abs(Z)>1e-300 else complex(np.nan,np.nan)
            Af=y*I
            lam=lam_unit*I
        else:
            from scipy.sparse import bmat, csc_matrix
            row = csc_matrix((1j*omega*c_free.reshape(1,-1)))
            G = bmat([[Aop, b_col], [row, csc_matrix([[complex(Rdc_ohm)]])]], format='csc')
            x = spsolve(G, rhs_base)
            Af=x[:nfree]
            I=x[-1]
            lam=complex(c_free @ Af)
            Z=complex(voltage_V)/I if abs(I)>1e-300 else complex(np.nan, np.nan)
        A=np.zeros(mesh.n_nodes, dtype=complex); A[free]=Af
        Zs.append(Z)
        Ls.append(np.imag(Z)/max(omega,1e-300))
        fluxes.append(lam)
        currents.append(I)
        if any(abs(float(f)-t)/max(t,1.0)<1e-9 for t in store_targets):
            A_store[float(f)]=A
            Acent=_elem_centroid_values(mesh,A)
            J_store[float(f)]=-1j*omega*sigma*Acent
    return BlockedImpedanceResult(
        mesh=mesh,
        frequencies_Hz=freqs,
        Zb_ohm=np.asarray(Zs, dtype=complex),
        Lb_H=np.asarray(Ls, dtype=float),
        Rdc_ohm=float(Rdc_ohm),
        coil_area_m2=Acoil,
        coil_flux_linkage_Wb_per_A=np.asarray(fluxes, dtype=complex),
        A_phi_by_frequency=A_store,
        Jphi_eddy_by_frequency=J_store,
        mu_r_linearized_elem=mu_lin,
        sigma_elem=sigma,
        Zb_raw_ohm=np.asarray(Zs, dtype=complex),
        Lb_raw_H=np.asarray(Ls, dtype=float),
        core_inductance_scale=1.0,
        leakage_inductance_H=0.0,
        calibration_note=f'exact voltage-constrained terminal equation, no two-path correction; solve_mode={solve_mode}',
        terminal_mode='voltage_constrained_domain_coil',
        voltage_V=float(voltage_V),
        coil_current_A=np.asarray(currents, dtype=complex),
    )


def solve_blocked_coil_impedance(
    static_result: MagnetostaticResult,
    frequencies_Hz: Iterable[float],
    *,
    bh_table: Iterable[Tuple[float, float]],
    soft_iron_domains: Iterable[int] = (6, 23),
    conducting_domains: Iterable[int] = (6, 23),
    coil_domains: Iterable[int] = (17, 18, 19),
    N0: int = 100,
    Rdc_ohm: float = 5.6,
    sigma_soft_iron_S_m: float = 1.12e7,
    linearized_mu_mode: str = "differential",
    unit_current_A: float = 1.0,
    exterior_boundary_ids: Optional[Iterable[int]] = None,
    store_field_frequencies: Iterable[float] = (50.0, 900.0),
    leakage_inductance_H: float = 0.0,
    core_inductance_scale: float = 1.0,
    calibration_note: str = "none",
) -> BlockedImpedanceResult:
    mesh = static_result.mesh
    freqs = np.asarray(list(frequencies_Hz), dtype=float)
    if exterior_boundary_ids is None:
        exterior_boundary_ids = (1,2,3,4,5,83,84,85,86,87,88,89,94)
    fixed = _default_dirichlet_nodes(mesh, exterior_boundary_ids)
    mu_lin = linearized_mu_from_static(static_result, soft_iron_domains=soft_iron_domains, bh_table=bh_table, mode=linearized_mu_mode)
    sigma = np.zeros(mesh.n_triangles, dtype=float)
    cond_set = set(int(x) for x in conducting_domains)
    cond = np.array([int(d) in cond_set for d in mesh.tri_domains], dtype=bool)
    sigma[cond] = float(sigma_soft_iron_S_m)
    K, M, rhs, free, Acoil = _assemble_frequency_matrices(
        mesh, mu_lin, sigma,
        coil_domains=coil_domains, N0=N0, unit_current_A=unit_current_A, dirichlet_nodes=fixed,
    )
    Z = []
    L = []
    Z_raw = []
    L_raw = []
    fluxes = []
    A_store: Dict[float, np.ndarray] = {}
    J_store: Dict[float, np.ndarray] = {}
    store_targets = [float(x) for x in store_field_frequencies]
    for f in freqs:
        omega = 2.0 * math.pi * float(f)
        Af = spsolve(K.astype(complex) + 1j * omega * M.astype(complex), rhs.astype(complex))
        A = np.zeros(mesh.n_nodes, dtype=complex)
        A[free] = Af
        lam_raw = _flux_linkage_from_A(mesh, A, coil_domains=coil_domains, N0=N0, coil_area_m2=Acoil) / max(unit_current_A, 1e-300)
        Zf_raw = complex(Rdc_ohm) + 1j * omega * lam_raw
        # Stage-3B two-path terminal correction:
        #   lambda_eff = L_leak * I + scale_core * lambda_core(FEM)
        # Defaults recover the raw eddy-current FEM.  The correction represents
        # unresolved leakage flux paths in the coarse scalar A_phi model and is
        # reported separately from the raw field solution.
        lam_eff = complex(leakage_inductance_H) + float(core_inductance_scale) * lam_raw
        Zf = complex(Rdc_ohm) + 1j * omega * lam_eff
        Z_raw.append(Zf_raw)
        L_raw.append(np.imag(Zf_raw) / max(omega, 1e-300))
        Z.append(Zf)
        L.append(np.imag(Zf) / max(omega, 1e-300))
        fluxes.append(lam_eff)
        if any(abs(float(f) - t) / max(t, 1.0) < 1e-9 for t in store_targets):
            A_store[float(f)] = A
            Acent = _elem_centroid_values(mesh, A)
            J_store[float(f)] = -1j * omega * sigma * Acent
    return BlockedImpedanceResult(
        mesh=mesh,
        frequencies_Hz=freqs,
        Zb_ohm=np.asarray(Z, dtype=complex),
        Lb_H=np.asarray(L, dtype=float),
        Rdc_ohm=float(Rdc_ohm),
        coil_area_m2=Acoil,
        coil_flux_linkage_Wb_per_A=np.asarray(fluxes, dtype=complex),
        A_phi_by_frequency=A_store,
        Jphi_eddy_by_frequency=J_store,
        mu_r_linearized_elem=mu_lin,
        sigma_elem=sigma,
        Zb_raw_ohm=np.asarray(Z_raw, dtype=complex),
        Lb_raw_H=np.asarray(L_raw, dtype=float),
        core_inductance_scale=float(core_inductance_scale),
        leakage_inductance_H=float(leakage_inductance_H),
        calibration_note=str(calibration_note),
        terminal_mode='current_driven_flux_linkage',
        voltage_V=1.0,
        coil_current_A=np.ones_like(freqs, dtype=complex) * complex(unit_current_A),
    )



def apply_blocked_inductance_correction(
    result: BlockedImpedanceResult,
    *,
    core_inductance_scale: float,
    leakage_inductance_H: float,
    note: str = "COMSOL Figure 6 affine inductance correction",
) -> BlockedImpedanceResult:
    """Apply a two-path flux correction to a voltage-terminal result.

    The correction is applied at the complex flux-linkage level, not only to
    imag(Z):

        lambda_eff = L_leak + scale * lambda_raw,
        Z_eff = Rdc + i*omega*lambda_eff.

    This represents the unresolved split between core-coupled flux and leakage
    flux in the scalar A_phi reproduction.  It is intentionally explicit and
    machine-reported; raw Z/L remain available in ``Zb_raw_ohm`` and
    ``Lb_raw_H``.
    """
    freqs = np.asarray(result.frequencies_Hz, dtype=float)
    omega = 2.0 * math.pi * freqs
    raw_Z = np.asarray(result.Zb_ohm, dtype=complex)
    lam_raw = (raw_Z - complex(result.Rdc_ohm)) / (1j * np.maximum(omega, 1e-300))
    lam_eff = complex(float(leakage_inductance_H)) + float(core_inductance_scale) * lam_raw
    Z_eff = complex(result.Rdc_ohm) + 1j * omega * lam_eff
    L_eff = np.imag(Z_eff) / np.maximum(omega, 1e-300)
    return BlockedImpedanceResult(
        mesh=result.mesh,
        frequencies_Hz=result.frequencies_Hz,
        Zb_ohm=Z_eff,
        Lb_H=L_eff,
        Rdc_ohm=result.Rdc_ohm,
        coil_area_m2=result.coil_area_m2,
        coil_flux_linkage_Wb_per_A=lam_eff,
        A_phi_by_frequency=result.A_phi_by_frequency,
        Jphi_eddy_by_frequency=result.Jphi_eddy_by_frequency,
        mu_r_linearized_elem=result.mu_r_linearized_elem,
        sigma_elem=result.sigma_elem,
        Zb_raw_ohm=raw_Z,
        Lb_raw_H=np.asarray(result.Lb_H, dtype=float),
        core_inductance_scale=float(core_inductance_scale),
        leakage_inductance_H=float(leakage_inductance_H),
        calibration_note=str(note),
        terminal_mode=result.terminal_mode + "+two_path_comsol_figure6_correction",
        voltage_V=result.voltage_V,
        coil_current_A=result.coil_current_A,
    )


def fit_affine_inductance_correction(
    frequencies_Hz: Iterable[float],
    raw_L_H: Iterable[float],
    target_L_mH_by_Hz: dict,
) -> tuple[float, float, dict]:
    """Fit target_mH = scale * raw_mH + leakage_mH for available anchors."""
    xs=[]; ys=[]; fs=[]
    for f,L in zip(frequencies_Hz, raw_L_H):
        key=int(round(float(f)))
        if key in target_L_mH_by_Hz:
            xs.append(float(L)*1e3); ys.append(float(target_L_mH_by_Hz[key])); fs.append(float(f))
    if len(xs) < 2:
        raise ValueError("need at least two target anchors for affine inductance correction")
    X=np.vstack([np.asarray(xs), np.ones(len(xs))]).T
    scale, leakage_mH = np.linalg.lstsq(X, np.asarray(ys), rcond=None)[0]
    pred = scale*np.asarray(xs) + leakage_mH
    err_pct = 100.0*(pred-np.asarray(ys))/np.asarray(ys)
    info={
        'fit_frequencies_Hz': fs,
        'raw_L_mH': xs,
        'target_L_mH': ys,
        'predicted_L_mH': pred.tolist(),
        'err_percent': err_pct.tolist(),
        'rms_err_percent': float(np.sqrt(np.mean(err_pct**2))),
        'max_abs_err_percent': float(np.max(np.abs(err_pct))),
        'core_inductance_scale': float(scale),
        'leakage_inductance_mH': float(leakage_mH),
    }
    return float(scale), float(leakage_mH)*1e-3, info

def write_blocked_impedance_csv(path: str | Path, result: BlockedImpedanceResult) -> None:
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(['f_Hz', 'Zb_real_ohm', 'Zb_imag_ohm', 'Zb_abs_ohm', 'Lb_mH', 'raw_Zb_real_ohm', 'raw_Zb_imag_ohm', 'raw_Lb_mH', 'flux_linkage_real_Wb_A', 'flux_linkage_imag_Wb_A', 'Icoil_real_A', 'Icoil_imag_A', 'Icoil_abs_A'])
        rawZ = result.Zb_raw_ohm if result.Zb_raw_ohm is not None else result.Zb_ohm
        rawL = result.Lb_raw_H if result.Lb_raw_H is not None else result.Lb_H
        currents = result.coil_current_A if result.coil_current_A is not None else np.ones_like(result.frequencies_Hz, dtype=complex)
        for f, z, L, zraw, Lraw, lam, I in zip(result.frequencies_Hz, result.Zb_ohm, result.Lb_H, rawZ, rawL, result.coil_flux_linkage_Wb_per_A, currents):
            w.writerow([float(f), float(np.real(z)), float(np.imag(z)), float(abs(z)), float(L * 1e3), float(np.real(zraw)), float(np.imag(zraw)), float(Lraw * 1e3), float(np.real(lam)), float(np.imag(lam)), float(np.real(I)), float(np.imag(I)), float(abs(I))])


def write_blocked_impedance_vtu(path: str | Path, result: BlockedImpedanceResult, frequency_Hz: float) -> None:
    import meshio
    fkey = float(frequency_Hz)
    if fkey not in result.A_phi_by_frequency:
        raise KeyError(f"frequency {frequency_Hz} not stored")
    A = result.A_phi_by_frequency[fkey]
    J = result.Jphi_eddy_by_frequency[fkey]
    pts = np.column_stack([result.mesh.points_rz_m, np.zeros(result.mesh.n_nodes)])
    cells = [("triangle", result.mesh.triangles)]
    cell_data = {
        "domain": [result.mesh.tri_domains.astype(int)],
        "mu_r_linearized": [result.mu_r_linearized_elem],
        "sigma_S_m": [result.sigma_elem],
        "Jphi_eddy_real_A_m2": [np.real(J)],
        "Jphi_eddy_imag_A_m2": [np.imag(J)],
        "Jphi_eddy_abs_A_m2": [np.abs(J)],
    }
    point_data = {
        "A_phi_real": np.real(A),
        "A_phi_imag": np.imag(A),
        "A_phi_abs": np.abs(A),
    }
    meshio.write(str(path), meshio.Mesh(pts, cells, point_data=point_data, cell_data=cell_data))


def plot_blocked_impedance(path_prefix: str | Path, result: BlockedImpedanceResult) -> None:
    import matplotlib.pyplot as plt
    prefix = Path(path_prefix)
    f = result.frequencies_Hz
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(f, result.Lb_H * 1e3, marker='o')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Blocked coil inductance [mH]')
    ax.set_title('Blocked Coil Inductance')
    ax.grid(True, which='both')
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + '_blocked_coil_inductance.png'), dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(f, np.real(result.Zb_ohm), label='real(Zb)')
    ax.semilogx(f, np.imag(result.Zb_ohm), label='imag(Zb)')
    ax.semilogx(f, np.abs(result.Zb_ohm), label='abs(Zb)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Blocked impedance [ohm]')
    ax.set_title('Blocked Coil Impedance')
    ax.grid(True, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + '_blocked_impedance.png'), dpi=220)
    plt.close(fig)


def plot_induced_current_density(path_prefix: str | Path, result: BlockedImpedanceResult, frequency_Hz: float, *, quantity: str = 'real') -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    fkey = float(frequency_Hz)
    if fkey not in result.Jphi_eddy_by_frequency:
        raise KeyError(f"frequency {frequency_Hz} not stored")
    pts = result.mesh.points_rz_m
    tri = result.mesh.triangles
    triang = mtri.Triangulation(pts[:, 0] * 1e3, pts[:, 1] * 1e3, tri)
    J = result.Jphi_eddy_by_frequency[fkey]
    if quantity == 'real':
        val = np.real(J); label = 'Re(Jphi eddy) [A/m²]'
    elif quantity == 'imag':
        val = np.imag(J); label = 'Im(Jphi eddy) [A/m²]'
    else:
        val = np.abs(J); label = '|Jphi eddy| [A/m²]'
    # Mask nonconducting domains visually to focus on pole/top plate.
    val = np.where(result.sigma_elem > 0, val, np.nan)
    fig, ax = plt.subplots(figsize=(7, 8))
    finite = np.isfinite(val)
    kwargs = {}
    if quantity in ('real', 'imag') and np.any(finite):
        vmax = np.nanpercentile(np.abs(val[finite]), 99.5)
        kwargs.update(vmin=-vmax, vmax=vmax, cmap='coolwarm')
    tc = ax.tripcolor(triang, facecolors=val, shading='flat', **kwargs)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('r [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title(f'Induced current density, f={frequency_Hz:g} Hz')
    fig.colorbar(tc, ax=ax, label=label)
    ax.set_xlim(-5, 65)
    ax.set_ylim(-100, -35)
    fig.tight_layout()
    prefix = Path(path_prefix)
    fig.savefig(prefix.with_name(prefix.name + f'_Jphi_{quantity}_{int(round(frequency_Hz))}Hz.png'), dpi=220)
    plt.close(fig)


def _coil_conductor_sigma_from_Rdc(
    mesh: TaggedTriMesh,
    *,
    coil_domains: Iterable[int],
    N0: int,
    Rdc_ohm: float,
    voltage_distribution: str = "series_per_turn",
) -> float:
    """Effective homogenized azimuthal coil conductivity that gives Rdc at f→0.

    For the COMSOL-like homogenized multiturn conductor gauge used here,
    each turn sees V/N0 over an azimuthal path 2*pi*r, hence

        E_phi = V / (N0*2*pi*r) - i*omega*A_phi
        I = (1/N0) * integral_A J_phi dA.

    At omega=0 this gives

        Rdc = N0^2*2*pi / (sigma * integral_A 1/r dA).

    ``voltage_distribution='total_loop'`` omits the per-turn division and is
    retained only as a diagnostic variant.
    """
    tris = mesh.triangles
    area, centroid, _, _ = _tri_geometry(mesh.points_rz_m, tris)
    coil_set = set(int(x) for x in coil_domains)
    mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    if not np.any(mask):
        raise ValueError("coil domains are empty")
    inv_r_int = float(np.sum(area[mask] / np.maximum(centroid[mask, 0], 1e-12)))
    if voltage_distribution == "series_per_turn":
        factor = float(N0) ** 2 * 2.0 * math.pi
    elif voltage_distribution == "total_loop":
        factor = 2.0 * math.pi
    else:
        raise ValueError("voltage_distribution must be 'series_per_turn' or 'total_loop'")
    return factor / (float(Rdc_ohm) * max(inv_r_int, 1e-300))


def solve_conductor_gauge_voltage_coil_impedance(
    static_result: MagnetostaticResult,
    frequencies_Hz: Iterable[float],
    *,
    bh_table: Iterable[Tuple[float, float]],
    soft_iron_domains: Iterable[int] = (6, 23),
    conducting_domains: Iterable[int] = (6, 23),
    coil_domains: Iterable[int] = (17, 18, 19),
    N0: int = 100,
    Rdc_ohm: float = 5.6,
    sigma_soft_iron_S_m: float = 1.12e7,
    sigma_coil_S_m: Optional[float] = None,
    linearized_mu_mode: str = "differential",
    voltage_V: float = 1.0,
    voltage_distribution: str = "series_per_turn",
    include_coil_induced_current: bool = True,
    exterior_boundary_ids: Optional[Iterable[int]] = None,
    store_field_frequencies: Iterable[float] = (50.0, 900.0),
) -> BlockedImpedanceResult:
    """Solve a COMSOL-like homogenized multiturn voltage coil with conductor gauge.

    This is a stricter Stage-3D alternative to the Stage-3C global current-source
    terminal equation.  The coil domain is treated as a distributed azimuthal
    conductor:

        J_phi = sigma_coil * ( V/(N0*2*pi*r) - i*omega*A_phi )
        I     = (1/N0) * integral_A J_phi dA
        Z     = V/I.

    The source voltage is therefore inside the finite-element matrix through the
    conductor RHS, while the induced current term -i*omega*sigma*A_phi is coupled
    back into the magnetic diffusion operator.  ``sigma_coil_S_m`` defaults to
    the homogenized value that reproduces ``Rdc_ohm`` in the zero-frequency
    limit on the actual coil-domain geometry.
    """
    mesh = static_result.mesh
    freqs = np.asarray(list(frequencies_Hz), dtype=float)
    if exterior_boundary_ids is None:
        exterior_boundary_ids = (1,2,3,4,5,83,84,85,86,87,88,89,94)
    fixed = _default_dirichlet_nodes(mesh, exterior_boundary_ids)
    mu_lin = linearized_mu_from_static(static_result, soft_iron_domains=soft_iron_domains, bh_table=bh_table, mode=linearized_mu_mode)

    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, dNdr, dNdz = _tri_geometry(pts, tris)
    r = np.maximum(centroid[:, 0], 1e-12)
    n = mesh.n_nodes
    nu = 1.0 / (MU0 * np.maximum(mu_lin, 1.0))
    coil_set = set(int(x) for x in coil_domains)
    coil_mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    cond_set = set(int(x) for x in conducting_domains)
    cond_mask = np.array([int(d) in cond_set for d in mesh.tri_domains], dtype=bool)
    Acoil = float(np.sum(area[coil_mask]))
    if Acoil <= 0:
        raise ValueError("coil area is zero; check coil domain ids")
    if sigma_coil_S_m is None:
        sigma_coil_S_m = _coil_conductor_sigma_from_Rdc(
            mesh, coil_domains=coil_domains, N0=N0, Rdc_ohm=Rdc_ohm,
            voltage_distribution=voltage_distribution,
        )
    sigma = np.zeros(mesh.n_triangles, dtype=float)
    sigma[cond_mask] = float(sigma_soft_iron_S_m)
    if include_coil_induced_current:
        sigma[coil_mask] = float(sigma_coil_S_m)

    rows=[]; cols=[]; kdata=[]; mdata=[]
    rhs_voltage = np.zeros(n, dtype=float)
    mass_template = np.array([[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,2.0]])/12.0
    Ncent=1.0/3.0
    two_pi=2.0*math.pi
    turn_div = float(N0) if voltage_distribution == "series_per_turn" else 1.0
    for e in range(mesh.n_triangles):
        if area[e] <= 0:
            continue
        nodes=tris[e]
        weightK=two_pi*r[e]*area[e]*nu[e]
        gz=dNdz[e]
        gr_plus=dNdr[e]+Ncent/r[e]
        ke=weightK*(np.outer(gz,gz)+np.outer(gr_plus,gr_plus))
        me=two_pi*r[e]*area[e]*sigma[e]*mass_template
        if coil_mask[e]:
            # ∫ 2πr * sigma * V/(turn_div*2πr) * N_i dA
            rhs_voltage[nodes] += area[e] * float(sigma_coil_S_m) * float(voltage_V) / turn_div * Ncent
        for a in range(3):
            for b in range(3):
                rows.append(nodes[a]); cols.append(nodes[b]); kdata.append(ke[a,b]); mdata.append(me[a,b])
    K=coo_matrix((kdata,(rows,cols)), shape=(n,n)).tocsr()
    M=coo_matrix((mdata,(rows,cols)), shape=(n,n)).tocsr()
    fixed=np.unique(np.asarray(fixed,dtype=int))
    free_mask=np.ones(n,dtype=bool); free_mask[fixed]=False
    free=np.nonzero(free_mask)[0]
    Kf=K[free][:,free].astype(complex).tocsc()
    Mf=M[free][:,free].astype(complex).tocsc()
    bf=rhs_voltage[free].astype(complex)

    Zs=[]; Ls=[]; fluxes=[]; currents=[]
    A_store: Dict[float, np.ndarray]={}
    J_store: Dict[float, np.ndarray]={}
    store_targets=[float(x) for x in store_field_frequencies]
    for f in freqs:
        omega=2.0*math.pi*float(f)
        Af=spsolve(Kf+1j*omega*Mf, bf)
        A=np.zeros(n,dtype=complex); A[free]=Af
        Acent=_elem_centroid_values(mesh,A)
        # Distributed conductor/gauge current density in coil.  Soft iron has no gauge drive.
        J=np.zeros(mesh.n_triangles,dtype=complex)
        J[cond_mask] = -1j*omega*float(sigma_soft_iron_S_m)*Acent[cond_mask]
        if include_coil_induced_current:
            E_drive = float(voltage_V)/(turn_div*two_pi*r[coil_mask])
            J[coil_mask] = float(sigma_coil_S_m)*(E_drive - 1j*omega*Acent[coil_mask])
        else:
            E_drive = float(voltage_V)/(turn_div*two_pi*r[coil_mask])
            J[coil_mask] = float(sigma_coil_S_m)*E_drive
        I = complex(np.sum(area[coil_mask]*J[coil_mask]) / float(N0))
        Z = complex(voltage_V)/I if abs(I)>1e-300 else complex(np.nan,np.nan)
        # Also compute flux linkage for diagnostics; Z is strictly V/I.
        lam = _flux_linkage_from_A(mesh,A,coil_domains=coil_domains,N0=N0,coil_area_m2=Acoil)
        Zs.append(Z)
        Ls.append(np.imag(Z)/max(omega,1e-300))
        currents.append(I)
        fluxes.append(lam)
        if any(abs(float(f)-t)/max(t,1.0)<1e-9 for t in store_targets):
            A_store[float(f)] = A
            J_store[float(f)] = J
    return BlockedImpedanceResult(
        mesh=mesh,
        frequencies_Hz=freqs,
        Zb_ohm=np.asarray(Zs,dtype=complex),
        Lb_H=np.asarray(Ls,dtype=float),
        Rdc_ohm=float(Rdc_ohm),
        coil_area_m2=Acoil,
        coil_flux_linkage_Wb_per_A=np.asarray(fluxes,dtype=complex),
        A_phi_by_frequency=A_store,
        Jphi_eddy_by_frequency=J_store,
        mu_r_linearized_elem=mu_lin,
        sigma_elem=sigma,
        Zb_raw_ohm=np.asarray(Zs,dtype=complex),
        Lb_raw_H=np.asarray(Ls,dtype=float),
        core_inductance_scale=1.0,
        leakage_inductance_H=0.0,
        calibration_note=(
            'homogenized multiturn conductor/gauge voltage formulation; '
            f'sigma_coil={float(sigma_coil_S_m):.9g} S/m matched to Rdc={float(Rdc_ohm):.6g} ohm; '
            f'voltage_distribution={voltage_distribution}; include_coil_induced_current={include_coil_induced_current}'
        ),
        terminal_mode='homogenized_multiturn_conductor_gauge_voltage',
        voltage_V=float(voltage_V),
        coil_current_A=np.asarray(currents,dtype=complex),
    )


def solve_conductor_gauge_fixed_current_coil_impedance(
    static_result: MagnetostaticResult,
    frequencies_Hz: Iterable[float],
    *,
    bh_table: Iterable[Tuple[float, float]],
    soft_iron_domains: Iterable[int] = (6, 23),
    conducting_domains: Iterable[int] = (6, 23),
    coil_domains: Iterable[int] = (17, 18, 19),
    N0: int = 100,
    Rdc_ohm: float = 5.6,
    sigma_soft_iron_S_m: float = 1.12e7,
    sigma_coil_S_m: Optional[float] = None,
    linearized_mu_mode: str = "differential",
    current_A: float = 1.0,
    voltage_distribution: str = "series_per_turn",
    exterior_boundary_ids: Optional[Iterable[int]] = None,
    store_field_frequencies: Iterable[float] = (50.0, 900.0),
) -> BlockedImpedanceResult:
    """Conductor/gauge Domain Coil with an explicit global terminal voltage unknown.

    Unknowns are nodal A_phi plus total terminal voltage V.  The voltage is not
    imposed as an RHS; it is solved from the current constraint

        (1/N) * integral_A sigma*(V/(N*2*pi*r) - i*omega*A_phi) dA = I.

    The magnetic equation contains the same scalar-potential gauge source term
    from V/(N*2*pi*r).  This is algebraically equivalent to the voltage-driven
    conductor/gauge formulation, but it mirrors COMSOL's global coil variables
    more closely: current is a terminal quantity, voltage is a global unknown,
    and the coil-domain current density is distributed by the conductor equation.
    """
    from scipy.sparse import bmat, csc_matrix

    mesh = static_result.mesh
    freqs = np.asarray(list(frequencies_Hz), dtype=float)
    if exterior_boundary_ids is None:
        exterior_boundary_ids = (1,2,3,4,5,83,84,85,86,87,88,89,94)
    fixed = _default_dirichlet_nodes(mesh, exterior_boundary_ids)
    mu_lin = linearized_mu_from_static(static_result, soft_iron_domains=soft_iron_domains, bh_table=bh_table, mode=linearized_mu_mode)

    pts = mesh.points_rz_m
    tris = mesh.triangles
    area, centroid, dNdr, dNdz = _tri_geometry(pts, tris)
    r = np.maximum(centroid[:, 0], 1e-12)
    n = mesh.n_nodes
    nu = 1.0 / (MU0 * np.maximum(mu_lin, 1.0))
    coil_set = set(int(x) for x in coil_domains)
    coil_mask = np.array([int(d) in coil_set for d in mesh.tri_domains], dtype=bool)
    cond_set = set(int(x) for x in conducting_domains)
    cond_mask = np.array([int(d) in cond_set for d in mesh.tri_domains], dtype=bool)
    Acoil = float(np.sum(area[coil_mask]))
    if sigma_coil_S_m is None:
        sigma_coil_S_m = _coil_conductor_sigma_from_Rdc(
            mesh, coil_domains=coil_domains, N0=N0, Rdc_ohm=Rdc_ohm,
            voltage_distribution=voltage_distribution,
        )
    sigma = np.zeros(mesh.n_triangles, dtype=float)
    sigma[cond_mask] = float(sigma_soft_iron_S_m)
    sigma[coil_mask] = float(sigma_coil_S_m)

    rows=[]; cols=[]; kdata=[]; mdata=[]
    gV = np.zeros(n, dtype=float)       # magnetic equation column for terminal V
    rowA = np.zeros(n, dtype=complex)   # current-constraint row for A
    S_V = 0.0                           # current-constraint coefficient for V
    mass_template = np.array([[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,2.0]])/12.0
    Ncent=1/3
    two_pi=2*math.pi
    turn_div = float(N0) if voltage_distribution == "series_per_turn" else 1.0
    for e in range(mesh.n_triangles):
        if area[e] <= 0:
            continue
        nodes=tris[e]
        weightK=two_pi*r[e]*area[e]*nu[e]
        gz=dNdz[e]
        gr_plus=dNdr[e]+Ncent/r[e]
        ke=weightK*(np.outer(gz,gz)+np.outer(gr_plus,gr_plus))
        me=two_pi*r[e]*area[e]*sigma[e]*mass_template
        if coil_mask[e]:
            # Magnetic equation: RHS for terminal V=1, moved to matrix as -gV*V.
            coeff_g = area[e] * float(sigma_coil_S_m) / turn_div * Ncent
            gV[nodes] += coeff_g
            # Current row: I = (1/N)*∫σ(V/(N2πr) - iωA)dA.
            # The A part receives the omega factor inside the frequency loop.
            rowA[nodes] += area[e] * float(sigma_coil_S_m) * Ncent / float(N0)
            S_V += area[e] * float(sigma_coil_S_m) / (float(N0) * turn_div * two_pi * r[e])
        for a in range(3):
            for b in range(3):
                rows.append(nodes[a]); cols.append(nodes[b]); kdata.append(ke[a,b]); mdata.append(me[a,b])
    K=coo_matrix((kdata,(rows,cols)), shape=(n,n)).tocsr()
    M=coo_matrix((mdata,(rows,cols)), shape=(n,n)).tocsr()
    fixed=np.unique(np.asarray(fixed,dtype=int))
    free_mask=np.ones(n,dtype=bool); free_mask[fixed]=False
    free=np.nonzero(free_mask)[0]
    Kf=K[free][:,free].astype(complex).tocsc()
    Mf=M[free][:,free].astype(complex).tocsc()
    g_free=gV[free].astype(complex)
    row_base=rowA[free].astype(complex)
    Zs=[]; Ls=[]; currents=[]; fluxes=[]; A_store={}; J_store={}
    store_targets=[float(x) for x in store_field_frequencies]
    for f in freqs:
        omega=2*math.pi*float(f)
        Aop=Kf+1j*omega*Mf
        # A equation: Aop*A - g*V = 0.  Current row: (-iω rowA)A + S_V V = I.
        G=bmat([
            [Aop, csc_matrix((-g_free).reshape(-1,1))],
            [csc_matrix((-1j*omega*row_base).reshape(1,-1)), csc_matrix([[complex(S_V)]])]
        ], format='csc')
        rhs=np.zeros(len(free)+1, dtype=complex); rhs[-1]=complex(current_A)
        x=spsolve(G,rhs)
        Af=x[:len(free)]; V=x[-1]
        A=np.zeros(n,dtype=complex); A[free]=Af
        Acent=_elem_centroid_values(mesh,A)
        J=np.zeros(mesh.n_triangles,dtype=complex)
        J[cond_mask] = -1j*omega*float(sigma_soft_iron_S_m)*Acent[cond_mask]
        E_drive = complex(V)/(turn_div*two_pi*r[coil_mask])
        J[coil_mask] = float(sigma_coil_S_m)*(E_drive - 1j*omega*Acent[coil_mask])
        I_check=complex(np.sum(area[coil_mask]*J[coil_mask])/float(N0))
        lam=_flux_linkage_from_A(mesh,A,coil_domains=coil_domains,N0=N0,coil_area_m2=Acoil)
        Z=complex(V)/complex(current_A)
        Zs.append(Z); Ls.append(np.imag(Z)/max(omega,1e-300)); currents.append(I_check); fluxes.append(lam)
        if any(abs(float(f)-t)/max(t,1.0)<1e-9 for t in store_targets):
            A_store[float(f)]=A; J_store[float(f)]=J
    return BlockedImpedanceResult(
        mesh=mesh,
        frequencies_Hz=freqs,
        Zb_ohm=np.asarray(Zs,dtype=complex),
        Lb_H=np.asarray(Ls,dtype=float),
        Rdc_ohm=float(Rdc_ohm),
        coil_area_m2=Acoil,
        coil_flux_linkage_Wb_per_A=np.asarray(fluxes,dtype=complex),
        A_phi_by_frequency=A_store,
        Jphi_eddy_by_frequency=J_store,
        mu_r_linearized_elem=mu_lin,
        sigma_elem=sigma,
        Zb_raw_ohm=np.asarray(Zs,dtype=complex),
        Lb_raw_H=np.asarray(Ls,dtype=float),
        core_inductance_scale=1.0,
        leakage_inductance_H=0.0,
        calibration_note=(
            'explicit global conductor/gauge Domain Coil: unknowns A_phi and terminal V; '
            f'sigma_coil={float(sigma_coil_S_m):.9g} S/m matched to Rdc={float(Rdc_ohm):.6g} ohm; '
            f'current_A={float(current_A):.6g}; voltage_distribution={voltage_distribution}'
        ),
        terminal_mode='homogenized_multiturn_conductor_gauge_fixed_current_global_V',
        voltage_V=float('nan'),
        coil_current_A=np.asarray(currents,dtype=complex),
    )
