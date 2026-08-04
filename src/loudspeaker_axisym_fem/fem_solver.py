from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple
import json
import numpy as np
import meshio
from scipy.sparse.linalg import spsolve
from scipy import special
from skfem import MeshTri, Basis, FacetBasis, ElementTriP1, ElementTriP2, BilinearForm, LinearForm, asm
from skfem.helpers import grad

from .meshgen import MeshTags, generate_gmsh_mesh


@dataclass(frozen=True)
class LoudspeakerParams:
    fmax: float = 10000.0
    c0: float = 343.0
    rho0: float = 1.2
    M_MD: float = 33.4e-3
    C_MS: float = 1.18e-3
    R_MS: float = 1.85
    BL: float = 11.4
    R_E: float = 7.0
    L_e: float = 6.89e-3
    n_e: float = 0.7
    V0: float = np.sqrt(2.0)
    R_g: float = 0.0
    a: float = 0.12
    Rair: float = 0.20
    Rpml: float = 0.06
    PMLfactor: float = 0.5
    PMLgamma: float = 5.0
    PMLstrength: float = 4.0
    p_ref: float = 20e-6

    @property
    def S_D(self) -> float:
        return np.pi * self.a**2
    @property
    def M_MS(self) -> float:
        return self.M_MD + 2.0 * self.S_D**2 * 8.0 * self.rho0 / (3.0 * np.pi**2 * self.a)
    @property
    def Fs(self) -> float:
        return 1.0 / (2.0 * np.pi * np.sqrt(self.C_MS * self.M_MS))
    @property
    def Q_ES(self) -> float:
        return 2.0 * np.pi * self.Fs * self.M_MS * self.R_E / self.BL**2
    @property
    def Q_MS(self) -> float:
        return 2.0 * np.pi * self.Fs * self.M_MS / self.R_MS
    @property
    def Q_TS(self) -> float:
        return self.Q_MS * self.Q_ES / (self.Q_MS + self.Q_ES)
    @property
    def V_AS(self) -> float:
        return self.rho0 * self.c0**2 * self.S_D**2 * self.C_MS
    @property
    def eta0(self) -> float:
        return 4.0 * np.pi**2 / self.c0**3 * self.Fs**3 * self.V_AS / self.Q_ES

    def write_json(self, path: str | Path) -> None:
        d = asdict(self)
        d.update(S_D=self.S_D, M_MS=self.M_MS, Fs=self.Fs, Q_ES=self.Q_ES,
                 Q_MS=self.Q_MS, Q_TS=self.Q_TS, V_AS=self.V_AS, eta0=self.eta0)
        Path(path).write_text(json.dumps(d, indent=2), encoding='utf-8')


def iso_1_24_frequencies() -> np.ndarray:
    return np.array([10, 10.3, 10.6, 10.9, 11.2, 11.5, 11.8, 12.2, 12.5, 12.8, 13.2, 13.6,
        14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.6,
        21.2, 21.8, 22.4, 23, 23.6, 24.3, 25, 25.8, 26.5, 27.2, 28, 29, 30,
        30.7, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.7, 40, 41.2, 42.5,
        43.7, 45, 46.2, 47.5, 48.7, 50, 51.5, 53, 54.5, 56, 58, 60, 61.5, 63,
        65, 67, 69, 71, 73, 75, 77.5, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5,
        100, 103, 106, 109, 112, 115, 118, 122, 125, 128, 132, 136, 140, 145,
        150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 206, 212, 218,
        224, 230, 236, 243, 250, 258, 265, 272, 280, 290, 300, 307, 315, 325,
        335, 345, 355, 365, 375, 387, 400, 412, 425, 437, 450, 462, 475, 487,
        500, 515, 530, 545, 560, 580, 600, 615, 630, 650, 670, 690, 710, 730,
        750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1e3, 1.03e3, 1.06e3,
        1.09e3, 1.12e3, 1.15e3, 1.18e3, 1.22e3, 1.25e3, 1.28e3, 1.32e3,
        1.36e3, 1.4e3, 1.45e3, 1.5e3, 1.55e3, 1.6e3, 1.65e3, 1.7e3, 1.75e3,
        1.8e3, 1.85e3, 1.9e3, 1.95e3, 2e3, 2.06e3, 2.12e3, 2.18e3, 2.24e3,
        2.3e3, 2.36e3, 2.43e3, 2.5e3, 2.58e3, 2.65e3, 2.72e3, 2.8e3, 2.9e3,
        3e3, 3.07e3, 3.15e3, 3.25e3, 3.35e3, 3.45e3, 3.55e3, 3.65e3,
        3.75e3, 3.87e3, 4e3, 4.12e3, 4.25e3, 4.37e3, 4.5e3, 4.62e3,
        4.75e3, 4.87e3, 5e3, 5.15e3, 5.3e3, 5.45e3, 5.6e3, 5.8e3, 6e3,
        6.15e3, 6.3e3, 6.5e3, 6.7e3, 6.9e3, 7.1e3, 7.3e3, 7.5e3, 7.75e3,
        8e3, 8.25e3, 8.5e3, 8.75e3, 9e3, 9.25e3, 9.5e3, 9.75e3, 1e4], dtype=float)


def coil_impedance(f: float, p: LoudspeakerParams) -> complex:
    w = 2 * np.pi * f
    L_E = (p.L_e / np.sin(p.n_e * np.pi / 2.0)) * w ** (p.n_e - 1.0)
    Rp_E = (p.L_e / np.cos(p.n_e * np.pi / 2.0)) * w ** p.n_e
    Zpar = 1.0 / (1.0 / (1j * w * L_E) + 1.0 / Rp_E)
    return p.R_g + p.R_E + Zpar


def mech_structural_impedance(f: float, p: LoudspeakerParams) -> complex:
    w = 2*np.pi*f
    return p.R_MS + 1j*w*p.M_MD + 1.0/(1j*w*p.C_MS)


def analytical_vars(f: np.ndarray, u: np.ndarray, p: LoudspeakerParams) -> Dict[str, np.ndarray]:
    w = 2*np.pi*f
    omega_s = 2*np.pi*p.Fs
    R_AE = p.BL**2/(p.S_D**2*p.R_E)
    R_AT = R_AE + p.R_MS/(p.S_D**2)
    s = 1j*w/omega_s
    Qlf_D = p.S_D*p.V0/p.BL*R_AE/R_AT*((1/p.Q_TS)*s)/(s**2+(1/p.Q_TS)*s+1)
    L_E = (p.L_e / np.sin(p.n_e*np.pi/2))*w**(p.n_e-1)
    omega_u1 = p.M_MS*p.R_E/(p.M_MD*L_E)
    Qhf_D = Qlf_D/(1+1j*w/omega_u1)
    k = w/p.c0
    x = np.maximum(2*k*p.a, 1e-12)
    P_AR_ana = (np.pi/2)*p.a**2*np.abs(u)**2*p.rho0*p.c0*(1-2*special.j1(x)/x)
    prms = p.rho0*w*np.sqrt(0.5*np.abs(u)**2)*p.S_D/(2*np.pi*1.0)
    return dict(Qlf_D=Qlf_D, Qhf_D=Qhf_D, P_AR_ana=P_AR_ana, prms=prms)


class AxisymmetricFEM:
    def __init__(self, msh_path: str | Path, params: LoudspeakerParams | None = None, element_order: int = 1):
        self.params = params or LoudspeakerParams()
        if element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")
        self.element_order = int(element_order)
        self.tags = MeshTags()
        mm = meshio.read(str(msh_path))
        pts2 = mm.points[:, :2]
        tri_blocks = [c.data for c in mm.cells if c.type == 'triangle']
        if not tri_blocks:
            raise ValueError('No triangle cells found in Gmsh mesh.')
        tris = np.vstack(tri_blocks)
        self.mesh = MeshTri(pts2.T, tris.T)
        self.element = ElementTriP2() if self.element_order == 2 else ElementTriP1()
        self.basis = Basis(self.mesh, self.element)
        self.N = self.basis.N
        self.boundary_facets = self._extract_facets(mm)
        self.outer_dofs = self.basis.get_dofs(facets=self.boundary_facets[self.tags.outer]).all()
        self.free_dofs = np.setdiff1d(np.arange(self.N), self.outer_dofs)

    def _extract_facets(self, mm) -> dict[int, np.ndarray]:
        facet_map = {tuple(sorted(self.mesh.facets[:, i])): i for i in range(self.mesh.facets.shape[1])}
        out: dict[int, list[int]] = {}
        for block, tags in zip(mm.cells, mm.cell_data['gmsh:physical']):
            if block.type != 'line':
                continue
            for seg, tag in zip(block.data, tags):
                key = tuple(sorted(map(int, seg)))
                if key in facet_map:
                    out.setdefault(int(tag), []).append(facet_map[key])
        return {k: np.array(sorted(set(v)), dtype=int) for k, v in out.items()}

    def _pml_coeffs_form(self, f: float):
        p = self.params
        k0 = 2*np.pi*f/p.c0
        # weak form for axisymmetric Helmholtz with approximate rational radial PML.
        @BilinearForm(dtype=complex)
        def helm(u, v, w):
            r = w.x[0]
            z = w.x[1]
            rad = np.sqrt(r*r + z*z) + 1e-14
            eta = np.clip((rad - p.Rair)/p.Rpml, 0.0, 0.999)
            # Rational-like stretch: small at interface, very high near outer edge; capped for conditioning.
            sig = p.PMLstrength * p.PMLfactor * eta**p.PMLgamma / (1.0 - eta + 0.08)
            sig = np.minimum(sig, 40.0)
            s_r = 1.0 - 1j*sig
            # Tangential stretch approximated by integrated radial stretch divided by radius.
            sig_t = p.PMLstrength * p.PMLfactor * eta**(p.PMLgamma + 1.0) / (1.0 - eta + 0.08)
            sig_t = np.minimum(sig_t, 40.0)
            s_t = 1.0 - 1j*sig_t
            s_pml = 1.0 - 1j*sig
            # scalar rational PML/sponge form; it is less exact than the tensor stretch but
            # robust for this compact axisymmetric tutorial geometry.
            gu = grad(u); gv = grad(v)
            stiffness = (gu[0]*gv[0] + gu[1]*gv[1]) / s_pml
            return 2*np.pi*r*(stiffness - k0*k0*s_pml*u*v)
        return helm

    def assemble_acoustics(self, f: float):
        p = self.params
        w0 = 2*np.pi*f
        A = asm(self._pml_coeffs_form(f), self.basis).tocsr()
        # Speaker velocity RHS for unit axial velocity u_D = 1 m/s.
        def make_rhs(facets):
            fb = FacetBasis(self.mesh, self.element, facets=facets)
            @LinearForm(dtype=complex)
            def rhs(v, w):
                # ∂p/∂n = -iωρ (u_D e_z · n)
                return 2*np.pi*w.x[0] * (-1j*w0*p.rho0*w.n[1]) * v
            return asm(rhs, fb)
        b = np.zeros(self.N, dtype=complex)
        b_front = np.zeros(self.N, dtype=complex)
        b_back = np.zeros(self.N, dtype=complex)
        if self.tags.speaker_front in self.boundary_facets:
            b_front = make_rhs(self.boundary_facets[self.tags.speaker_front])
            b += b_front
        if self.tags.speaker_back in self.boundary_facets:
            b_back = make_rhs(self.boundary_facets[self.tags.speaker_back])
            b += b_back

        def make_force(facets):
            fb = FacetBasis(self.mesh, self.element, facets=facets)
            @LinearForm(dtype=complex)
            def force(v, w):
                # Fluid pressure contribution projected onto axial diaphragm motion.
                return 2*np.pi*w.x[0] * w.n[1] * v
            return asm(force, fb)
        c_front = np.zeros(self.N, dtype=complex)
        c_back = np.zeros(self.N, dtype=complex)
        if self.tags.speaker_front in self.boundary_facets:
            c_front = make_force(self.boundary_facets[self.tags.speaker_front])
        if self.tags.speaker_back in self.boundary_facets:
            c_back = make_force(self.boundary_facets[self.tags.speaker_back])
        c_total = c_front + c_back
        return A, b, c_total, c_front, c_back

    def solve_unit_acoustic(self, f: float, force_sign: float = 1.0):
        A, b, c_total, c_front, c_back = self.assemble_acoustics(f)
        I = self.free_dofs
        x = np.zeros(self.N, dtype=complex)
        x[I] = spsolve(A[I][:, I], b[I])
        # Force opposing the structural motion; sign is selected by calibration.
        Z_total = force_sign * np.dot(c_total, x)
        Z_front = force_sign * np.dot(c_front, x)
        Z_back = force_sign * np.dot(c_back, x)
        return x, Z_total, Z_front, Z_back

    def calibrate_force_sign(self, f: float = 100.0) -> float:
        x, Z, *_ = self.solve_unit_acoustic(f, force_sign=1.0)
        return 1.0 if np.real(Z) >= 0 else -1.0

    def solve_sweep(self, freqs: Iterable[float], force_sign: float | None = None, save_fields_at: Iterable[float] = (1000.0, 5000.0)) -> Dict[str, np.ndarray | dict]:
        p = self.params
        freqs = np.array(list(freqs), dtype=float)
        if force_sign is None:
            force_sign = self.calibrate_force_sign(float(freqs[min(len(freqs)//3, len(freqs)-1)]))
        n = len(freqs)
        u = np.zeros(n, dtype=complex)
        ic = np.zeros(n, dtype=complex)
        Zvoice = np.zeros(n, dtype=complex)
        Zac = np.zeros(n, dtype=complex)
        Zac_front = np.zeros(n, dtype=complex)
        Zac_back = np.zeros(n, dtype=complex)
        PE = np.zeros(n, dtype=float)
        PAR = np.zeros(n, dtype=float)
        fields: dict[float, np.ndarray] = {}
        field_targets = {float(t): None for t in save_fields_at}
        target_indices = {int(np.argmin(np.abs(freqs - t))): float(t) for t in field_targets}
        for j, f in enumerate(freqs):
            p_unit, Za, Zf, Zb = self.solve_unit_acoustic(float(f), force_sign=force_sign)
            Ze = coil_impedance(float(f), p)
            Zm = mech_structural_impedance(float(f), p)
            # V = Ze*i + BL*u; (Zm + Za)*u = BL*i
            u[j] = p.BL*p.V0 / (Ze*(Zm + Za) + p.BL**2)
            ic[j] = (Zm + Za)*u[j]/p.BL
            Zvoice[j] = p.V0/ic[j]
            Zac[j] = Za
            Zac_front[j] = Zf
            Zac_back[j] = Zb
            PE[j] = 0.5*np.real(p.V0*np.conj(ic[j]))
            PAR[j] = 0.5*np.maximum(np.real(Zf), 0.0)*abs(u[j])**2
            if j in target_indices:
                fields[float(freqs[j])] = p_unit * u[j]
        eta = PAR/np.maximum(PE, 1e-300)
        ana = analytical_vars(freqs, u, p)
        # Use piston/exterior 1 m approximation for SPL/phase from computed u_D.
        w = 2*np.pi*freqs; k = w/p.c0
        p1m = 1j*w*p.rho0*u*p.S_D*np.exp(-1j*k)/(2*np.pi)
        SPL = 20*np.log10(np.maximum(np.abs(p1m)/np.sqrt(2), 1e-300)/p.p_ref)
        phase = np.unwrap(np.angle(p1m/np.exp(-1j*k)))*180/np.pi
        return dict(f=freqs, u_D=u, i_c=ic, Zvoice=Zvoice, Zac=Zac, Zac_front=Zac_front, Zac_back=Zac_back,
                    P_E=PE, P_AR_front=PAR, eta=eta, SPL_1m=SPL, phase_rel_plane=phase,
                    force_sign=np.array([force_sign]), fields=fields, **ana)


def create_fem_model(workdir: str | Path, closed_back: bool, params: LoudspeakerParams | None = None,
                     h: float = 0.010, h_speaker: float = 0.0035):
    p = params or LoudspeakerParams()
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    msh_path = workdir / ('closed_back.msh' if closed_back else 'open_back.msh')
    if not msh_path.exists():
        generate_gmsh_mesh(msh_path, closed_back=closed_back, a=p.a, Rair=p.Rair, Rpml=p.Rpml,
                           h=h, h_speaker=h_speaker)
    return AxisymmetricFEM(msh_path, p)
