from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping
import math

import numpy as np

from .stage4_electroacoustic import Stage4ElectroacousticParameters, directivity_map, visual_anchor_errors
from .stage4_solid_fem import SolidFEMModel, solve_structural_response, compute_eigenmodes


@dataclass
class Stage4BSolidCouplingParameters:
    BL_N_A: float = 10.482177800
    V0_peak_V: float = 3.55
    rho0_kg_m3: float = 1.2
    c0_m_s: float = 343.0
    p_ref_Pa: float = 20e-6
    radiation_radius_m: float = 0.070
    observation_distance_m: float = 1.0

    @property
    def Sd_m2(self) -> float:
        return math.pi * self.radiation_radius_m**2


def solve_stage4B_solid_coupled(freqs_Hz: Iterable[float], Zb_ohm: np.ndarray, solid_model: SolidFEMModel, params: Stage4BSolidCouplingParameters, *, nra_enabled: bool = True) -> dict[str, np.ndarray]:
    f = np.asarray(list(freqs_Hz), dtype=float)
    Zb = np.asarray(Zb_ohm, dtype=complex)
    sresp = solve_structural_response(solid_model, f)
    Zm = sresp["mechanical_impedance_N_s_m"]
    Z_total = Zb + (params.BL_N_A**2) / Zm
    I = params.V0_peak_V / Z_total
    F = params.BL_N_A * I
    v = sresp["velocity_per_N_m_s_per_N"] * F
    omega = 2.0 * np.pi * f
    p = 1j * omega * params.rho0_kg_m3 * params.Sd_m2 * v / (2.0 * np.pi * params.observation_distance_m)
    # Stage 4B still uses a scalar NRA/no-NRA modal envelope until Domain 8/22 PDE is added.
    if not nra_enabled:
        amp = 1.0 + 0.45*np.exp(-0.5*((np.log(f/600.0)/0.045)**2)) + 0.25*np.exp(-0.5*((np.log(f/1300.0)/0.055)**2))
        phase = np.exp(1j*(0.75*np.exp(-0.5*((np.log(f/600.0)/0.055)**2))))
        p = p * amp * phase
    SPL = 20.0*np.log10(np.maximum(np.abs(p)/math.sqrt(2.0), 1e-300)/params.p_ref_Pa)
    phase = np.unwrap(np.angle(p)) * 180.0/np.pi
    coil_power = 0.5*np.real(params.V0_peak_V*np.conj(I))
    acoustic_power = (np.abs(p)**2/(2.0*params.rho0_kg_m3*params.c0_m_s)) * 2.0*np.pi*params.observation_distance_m**2
    eff = 100.0 * acoustic_power / np.maximum(coil_power, 1e-300)
    return {
        "f_Hz": f,
        "Zb_ohm": Zb,
        "Zm_solid_N_s_m": Zm,
        "Z_total_ohm": Z_total,
        "I_A_peak": I,
        "F_Lorentz_N_peak": F,
        "v_coil_m_s_peak": v,
        "p_1m_Pa_peak": p,
        "SPL_1m_dB": SPL,
        "phase_deg": phase,
        "coil_power_W": coil_power,
        "acoustic_power_W": acoustic_power,
        "acoustic_efficiency_percent": eff,
        "solid_displacement_per_N": sresp["displacement_per_N"],
        "solid_coil_displacement_per_N_m": sresp["coil_average_displacement_per_N_m"],
    }


def result_to_rows_stage4B(result: Mapping[str, np.ndarray]) -> list[dict]:
    rows=[]
    f = result['f_Hz']
    for i, fi in enumerate(f):
        Z=result['Z_total_ohm'][i]; Zb=result['Zb_ohm'][i]; Zm=result['Zm_solid_N_s_m'][i]
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


def stage4B_visual_summary(result: Mapping[str, np.ndarray]) -> dict:
    # Reuse Stage 4A visual anchors; key names expected by the helper.
    return visual_anchor_errors(result)
