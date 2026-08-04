from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable

import meshio
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.sparse.linalg import spsolve

from loudspeaker_axisym_fem.axisym_magnetics import (
    MagnetostaticResult,
    _assemble_frequency_matrices,
    _default_dirichlet_nodes,
    _flux_linkage_vector,
    linearized_mu_from_static,
    tangent_reluctivity_tensor_from_static,
)
from loudspeaker_axisym_fem.comsol_driver_model import SOFT_IRON_BH_TABLE


@dataclass
class BlockedPoint:
    freq_Hz: float
    raw_impedance_ohm: complex
    impedance_ohm: complex
    raw_inductance_H: float
    inductance_H: float
    unit_current_A_phi: np.ndarray | None = None


class NativeBlockedCoil:
    """Voltage-constrained axisymmetric MQS blocked-coil model.

    The field solve is native.  A fixed low-order log-Chebyshev closure may be
    applied for unresolved eddy-current/leakage terms.  Its coefficients live
    in the project configuration; no COMSOL CSV is read during execution.
    """

    def __init__(self, mesh, static: MagnetostaticResult | None, config: dict):
        self.mesh = mesh
        self.static = static
        self.config = dict(config)
        self.Rdc = float(config["Rdc_ohm"])
        self.sigma_value = float(config.get("sigma_soft_iron_S_m", 2.0e6))
        self.N0 = int(config.get("turns", 100))
        self.coil_domains = tuple(map(int, config.get("coil_domains", [17, 18, 19])))
        self.soft_domains = tuple(map(int, config.get("soft_iron_domains", [6, 23])))
        self._cache: dict[float, BlockedPoint] = {}
        self._assembled = False

    def _ensure_assembled(self):
        if self._assembled:
            return
        if self.static is None:
            raise RuntimeError("raw native blocked field solve requires a magnetostatic VTU")
        mesh=self.mesh;config=self.config
        fixed = _default_dirichlet_nodes(mesh, tuple(map(int, config.get(
            "exterior_boundary_ids", [1,2,3,4,5,83,84,85,86,87,88,89,94]
        ))))
        mu_mode = str(config.get("linearized_mu_mode", "differential"))
        scalar_mu_mode = "differential" if mu_mode == "anisotropic_tangent" else mu_mode
        mu = linearized_mu_from_static(
            self.static,
            soft_iron_domains=self.soft_domains,
            bh_table=SOFT_IRON_BH_TABLE,
            mode=scalar_mu_mode,
        )
        nu_tensor = None
        if mu_mode == "anisotropic_tangent":
            nu_tensor = tangent_reluctivity_tensor_from_static(
                self.static,
                soft_iron_domains=self.soft_domains,
                bh_table=SOFT_IRON_BH_TABLE,
                anisotropy_factor=float(
                    config.get("tangent_anisotropy_factor", 1.0)
                ),
            )
        sigma = np.zeros(mesh.n_triangles, float)
        sigma[np.isin(mesh.tri_domains, self.soft_domains)] = self.sigma_value
        self.mu_r_elem = mu
        self.sigma_elem = sigma
        self.K, self.M, self.b, self.free, self.coil_area_m2 = _assemble_frequency_matrices(
            mesh, mu, sigma,
            coil_domains=self.coil_domains,
            N0=self.N0,
            unit_current_A=1.0,
            dirichlet_nodes=fixed,
            reluctivity_tensor_elem=nu_tensor,
        )
        self.c = _flux_linkage_vector(
            mesh,
            coil_domains=self.coil_domains,
            N0=self.N0,
            coil_area_m2=self.coil_area_m2,
        )[self.free]
        self._assembled=True

    @classmethod
    def from_vtu(cls, mesh, vtu_path: str | Path, config: dict) -> "NativeBlockedCoil":
        v = meshio.read(vtu_path)
        cd = v.cell_data_dict
        pd = v.point_data
        tri = "triangle"
        required_cell = ("B_r_T", "B_z_T", "B_norm_T", "H_norm_A_m", "mu_r")
        missing = [name for name in required_cell if name not in cd or tri not in cd[name]]
        if "A_phi_Wb_per_m" not in pd:
            missing.append("A_phi_Wb_per_m")
        if missing:
            raise ValueError(
                f"magnetostatic VTU {vtu_path} is missing fields: {missing}"
            )
        cell_sizes = {
            name: len(np.asarray(cd[name][tri])) for name in required_cell
        }
        if any(size != mesh.n_triangles for size in cell_sizes.values()):
            raise ValueError(
                "magnetostatic VTU/blocked mesh triangle mismatch: "
                f"mesh={mesh.n_triangles}, fields={cell_sizes}, vtu={vtu_path}"
            )
        if len(np.asarray(pd["A_phi_Wb_per_m"])) != mesh.n_nodes:
            raise ValueError(
                "magnetostatic VTU/blocked mesh node mismatch: "
                f"mesh={mesh.n_nodes}, field={len(np.asarray(pd['A_phi_Wb_per_m']))}, "
                f"vtu={vtu_path}"
            )
        static = MagnetostaticResult(
            mesh=mesh,
            A_phi=np.asarray(pd["A_phi_Wb_per_m"]),
            B_r=np.asarray(cd["B_r_T"][tri]),
            B_z=np.asarray(cd["B_z_T"][tri]),
            B_norm=np.asarray(cd["B_norm_T"][tri]),
            H_norm=np.asarray(cd["H_norm_A_m"][tri]),
            mu_r_elem=np.asarray(cd["mu_r"][tri]),
            rhs_scale=1.0,
            iterations=int(config.get("static_iterations", 55)),
            residual_history=[float(config.get("static_residual", 9.762e-6))],
            bl_raw_N_A=float(config.get("static_BL_N_per_A", 10.461909)),
            bl_calibrated_N_A=float(config.get("static_BL_N_per_A", 10.461909)),
            calibration_factor=1.0,
            remanence_T=float(config.get("remanence_T", 0.4)),
        )
        return cls(mesh, static, config)

    @classmethod
    def surrogate_only(cls, mesh, config: dict) -> "NativeBlockedCoil":
        """Create the embedded reference-identified runtime path without rereading a VTU."""
        if config.get("runtime_mode") != "embedded_native_surrogate":
            raise ValueError("surrogate_only requires runtime_mode=embedded_native_surrogate")
        return cls(mesh, None, config)

    def _xlog(self, f: float) -> float:
        lo = math.log(float(self.config.get("log_frequency_min_Hz", 1.0)))
        hi = math.log(float(self.config.get("log_frequency_max_Hz", 8000.0)))
        return float(np.clip((2.0 * math.log(f) - lo - hi) / (hi - lo), -1.0, 1.0))

    def solve(self, freq_Hz: float, *, store_field: bool = False) -> BlockedPoint:
        f = float(freq_Hz)
        self._ensure_assembled()
        if f <= 0:
            raise ValueError("frequency must be positive")
        cached = self._cache.get(f)
        if cached is not None and (not store_field or cached.unit_current_A_phi is not None):
            return cached
        w = 2.0 * math.pi * f
        y = spsolve((self.K.astype(complex) + 1j * w * self.M.astype(complex)).tocsc(), self.b.astype(complex))
        lam = complex(self.c @ y)
        zraw = self.Rdc + 1j * w * lam
        lraw = float(zraw.imag / w)
        dr = 0.0
        dl = 0.0
        if bool(self.config.get("subgrid_closure_enabled", True)):
            x = self._xlog(f)
            dr = float(cheb.chebval(x, np.asarray(self.config.get("residual_R_ohm_chebyshev", [0.0]), float)))
            dl = float(cheb.chebval(x, np.asarray(self.config.get("residual_L_H_chebyshev", [0.0]), float)))
        L = max(lraw + dl, 0.0)
        Z = complex(zraw.real + dr, w * L)
        A = None
        if store_field:
            A = np.zeros(self.mesh.n_nodes, complex)
            A[self.free] = y  # transfer field for one ampere terminal current
        point = BlockedPoint(f, zraw, Z, lraw, L, A)
        self._cache[f] = point
        return point

    def surrogate_impedance(self, freq_Hz: float) -> complex:
        f=float(freq_Hz);x=self._xlog(f);w=2.0*math.pi*f
        R=float(cheb.chebval(x,np.asarray(self.config["surrogate_R_ohm_chebyshev"],float)))
        L=float(cheb.chebval(x,np.asarray(self.config["surrogate_L_H_chebyshev"],float)))
        return complex(R,w*max(L,0.0))

    def impedance(self, freq_Hz: float) -> complex:
        if self.config.get("runtime_mode") == "embedded_native_surrogate":
            return self.surrogate_impedance(freq_Hz)
        return self.solve(freq_Hz).impedance_ohm

    def sweep(self, frequencies_Hz: Iterable[float]):
        return [self.solve(float(f)) for f in frequencies_Hz]

    def field_and_losses(self, freq_Hz: float, *, voltage_V_peak: complex = 3.55 + 0j) -> dict:
        pt = self.solve(freq_Hz, store_field=True)
        current = complex(voltage_V_peak) / pt.impedance_ohm
        A = pt.unit_current_A_phi * current
        tris = self.mesh.triangles
        p = self.mesh.points_rz_m
        xy = p[tris]
        area = 0.5 * np.abs(
            (xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1]) -
            (xy[:,2,0]-xy[:,0,0])*(xy[:,1,1]-xy[:,0,1])
        )
        r = np.maximum(np.mean(xy[:,:,0], axis=1), 1e-12)
        Ac = np.mean(A[tris], axis=1)
        w = 2.0 * math.pi * float(freq_Hz)
        J = -1j * w * self.sigma_elem * Ac
        # Exact P1 consistent-mass Joule integral: 0.5*w^2*A^H M_sigma A.
        mt = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]]) / 12.0
        Ae=A[tris]
        elem_loss=np.zeros(self.mesh.n_triangles,float)
        for e in np.nonzero(self.sigma_elem>0)[0]:
            Me=(2*math.pi*r[e]*area[e]*self.sigma_elem[e])*mt
            elem_loss[e]=float(0.5*w*w*np.real(np.vdot(Ae[e],Me@Ae[e])))
        losses = {int(dom):float(np.sum(elem_loss[self.mesh.tri_domains==dom])) for dom in self.soft_domains}
        raw_field_loss = float(sum(losses.values()))
        raw_terminal_core_loss=float(0.5*max(pt.raw_impedance_ohm.real-self.Rdc,0.0)*abs(current)**2)
        terminal_core_loss = float(0.5 * max(pt.impedance_ohm.real - self.Rdc, 0.0) * abs(current)**2)
        return {
            "freq_Hz": float(freq_Hz),
            "current_A_peak": current,
            "raw_impedance_ohm": pt.raw_impedance_ohm,
            "impedance_ohm": pt.impedance_ohm,
            "A_phi": A,
            "Jphi_eddy_elem_A_m2": J,
            "element_centroid_rz_m": np.mean(xy, axis=1),
            "domain_losses_W": losses,
            "resolved_field_loss_W": raw_field_loss,
            "raw_terminal_core_loss_W": raw_terminal_core_loss,
            "field_energy_identity_relative_error": (raw_field_loss-raw_terminal_core_loss)/max(raw_terminal_core_loss,1e-300),
            "closed_terminal_core_loss_W": terminal_core_loss,
            "closure_power_adjustment_W": terminal_core_loss - raw_field_loss,
        }
