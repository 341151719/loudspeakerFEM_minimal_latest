from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence
import math

import numpy as np
from scipy import special


@dataclass(frozen=True)
class Stage4ElectroacousticParameters:
    """COMSOL loudspeaker tutorial Study-2 lumped electro-acoustic baseline.

    This is not the final solid/acoustic FEM replacement.  It is a physically
    explicit Stage-4A bridge that consumes Stage-2 BL and Stage-3C blocked
    impedance and reproduces the main Figure-8/Figure-10 scalar observables.
    """

    BL_N_A: float = 10.48
    V0_peak_V: float = 3.55
    rho0_kg_m3: float = 1.2041
    c0_m_s: float = 343.0
    p_ref_Pa: float = 20e-6
    radius_eval_m: float = 1.0
    effective_radius_m: float = 0.0700
    Mms_kg: float = 0.0120
    f0_Hz: float = 53.237
    Rms_N_s_m: float = 3.9333333333333336
    structural_loss_factor: float = 0.055
    first_breakup_Hz: float = 2350.0
    breakup_q: float = 6.0
    breakup_depth: float = 0.05
    high_frequency_rolloff_Hz: float = 16000.0
    high_frequency_rolloff_order: float = 2.0
    nra_mode1_Hz: float = 600.0
    nra_mode2_Hz: float = 1300.0
    nra_gain_complete_dB: float = 0.15
    nra_gain_lossless_dB: float = 4.8
    nra_q_complete: float = 3.0
    nra_q_lossless: float = 28.0

    @property
    def Sd_m2(self) -> float:
        return math.pi * self.effective_radius_m**2

    @property
    def Cms_m_N(self) -> float:
        w0 = 2.0 * math.pi * self.f0_Hz
        return 1.0 / (self.Mms_kg * w0*w0)


def _interp_complex_log_frequency(freqs_src: np.ndarray, values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    x = np.log(np.asarray(freqs_src, dtype=float))
    xi = np.log(np.asarray(freqs, dtype=float))
    re = np.interp(xi, x, np.real(values))
    im = np.interp(xi, x, np.imag(values))
    return re + 1j*im


def load_blocked_impedance_csv(path: str | Path, freqs: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    f0 = df['f_Hz'].to_numpy(float)
    z0 = df['Zb_real_ohm'].to_numpy(float) + 1j*df['Zb_imag_ohm'].to_numpy(float)
    if freqs is None:
        return f0, z0
    f = np.asarray(freqs, dtype=float)
    return f, _interp_complex_log_frequency(f0, z0, f)


def comsol_frequency_vector() -> np.ndarray:
    # COMSOL Study 1/2: 1..9 plus ISO preferred 1/12-octave 10 Hz to 8 kHz.
    return np.array([
        1,2,3,4,5,6,7,8,9,
        10,10.6,11.2,11.8,12.5,13.2,14,15,16,17,18,19,20,21.2,22.4,23.6,25,26.5,28,30,31.5,33.5,35.5,37.5,40,42.5,45,47.5,50,53,56,60,63,67,71,75,80,85,90,95,100,106,112,118,125,132,140,150,160,170,180,190,200,212,224,236,250,265,280,300,315,335,355,375,400,425,450,475,500,530,560,600,630,670,710,750,800,850,900,950,1000,1060,1120,1180,1250,1320,1400,1500,1600,1700,1800,1900,2000,2120,2240,2360,2500,2650,2800,3000,3150,3350,3550,3750,4000,4250,4500,4750,5000,5300,5600,6000,6300,6700,7100,7500,8000
    ], dtype=float)


def piston_radiation_impedance_mechanical(freqs_Hz: np.ndarray, p: Stage4ElectroacousticParameters) -> np.ndarray:
    f = np.asarray(freqs_Hz, dtype=float)
    k = 2.0*np.pi*f/p.c0_m_s
    a = p.effective_radius_m
    ka = np.maximum(k*a, 1e-12)
    # Average radiation impedance of a baffled circular piston, normalized by rho*c*S.
    R = 1.0 - special.j1(2.0*ka)/ka
    X = special.struve(1, 2.0*ka)/ka
    return p.rho0_kg_m3*p.c0_m_s*p.Sd_m2*(R + 1j*X)


def mechanical_impedance(freqs_Hz: np.ndarray, p: Stage4ElectroacousticParameters, include_radiation: bool = True) -> np.ndarray:
    f = np.asarray(freqs_Hz, dtype=float)
    w = 2*np.pi*f
    C = p.Cms_m_N
    # Add a small stiffness-proportional loss term so the phase trend is smooth.
    Z = p.Rms_N_s_m + 1j*w*p.Mms_kg + 1.0/(1j*w*C)
    Z += p.structural_loss_factor/(w*C)
    if include_radiation:
        Z += piston_radiation_impedance_mechanical(f, p)
    return Z


def breakup_complex_factor(freqs_Hz: np.ndarray, p: Stage4ElectroacousticParameters) -> np.ndarray:
    f = np.asarray(freqs_Hz, dtype=float)
    w = 2*np.pi*f
    wb = 2*np.pi*p.first_breakup_Hz
    H = (wb*wb)/(wb*wb - w*w + 1j*w*wb/p.breakup_q)
    # Mode is area-normalized in this Stage-4A baseline, so it mainly changes high-frequency phase/efficiency.
    roll = 1.0/np.sqrt(1.0 + (f/p.high_frequency_rolloff_Hz)**(2.0*p.high_frequency_rolloff_order))
    return roll*(1.0 - p.breakup_depth*H)


def nra_pressure_factor(freqs_Hz: np.ndarray, p: Stage4ElectroacousticParameters, *, enabled: bool = True) -> np.ndarray:
    f = np.asarray(freqs_Hz, dtype=float)
    out = np.ones_like(f, dtype=complex)
    gain_db = p.nra_gain_complete_dB if enabled else p.nra_gain_lossless_dB
    q = p.nra_q_complete if enabled else p.nra_q_lossless
    amp = 10.0**(gain_db/20.0) - 1.0
    for fm in (p.nra_mode1_Hz, p.nra_mode2_Hz):
        x = f/fm
        H = 1.0/(1.0 - x*x + 1j*x/q)
        out *= (1.0 + amp*H)
    return out


def solve_stage4_lumped(freqs_Hz: Sequence[float], Zb: Sequence[complex], p: Stage4ElectroacousticParameters, *, narrow_region_enabled: bool = True) -> dict:
    f = np.asarray(freqs_Hz, dtype=float)
    w = 2*np.pi*f
    Zb = np.asarray(Zb, dtype=complex)
    Zm = mechanical_impedance(f, p)
    Zem = p.BL_N_A*p.BL_N_A / Zm
    Ztot = Zb + Zem
    I = p.V0_peak_V / Ztot
    v = p.BL_N_A * I / Zm
    U = v * p.Sd_m2
    direct = breakup_complex_factor(f, p) * nra_pressure_factor(f, p, enabled=narrow_region_enabled)
    p_axis_peak = 1j*w*p.rho0_kg_m3*U/(2*np.pi*p.radius_eval_m) * direct
    p_axis_rms = np.abs(p_axis_peak)/math.sqrt(2.0)
    spl = 20*np.log10(np.maximum(p_axis_rms, 1e-300)/p.p_ref_Pa)
    phase_deg = np.unwrap(np.angle(p_axis_peak))*180/np.pi
    coil_power_W = 0.5*np.real(p.V0_peak_V*np.conj(I))
    acoustic_power_W = 0.5*p.rho0_kg_m3*p.c0_m_s*(np.abs(U)**2)*np.real(piston_radiation_impedance_mechanical(f,p)/(p.rho0_kg_m3*p.c0_m_s*p.Sd_m2))/max(p.Sd_m2,1e-30)
    # Safer estimate from piston radiation mechanical resistance: 0.5*Rrad*|v|^2
    Rrad = np.real(piston_radiation_impedance_mechanical(f, p))
    acoustic_power_W = 0.5*Rrad*np.abs(v)**2
    eff = np.where(coil_power_W > 1e-30, 100*acoustic_power_W/coil_power_W, np.nan)
    return {
        'f_Hz': f,
        'Zb_ohm': Zb,
        'Zmot_ohm': Zem,
        'Z_total_ohm': Ztot,
        'I_A_peak': I,
        'v_m_s_peak': v,
        'U_m3_s_peak': U,
        'p_axis_Pa_peak': p_axis_peak,
        'SPL_1m_dB': spl,
        'phase_deg': phase_deg,
        'coil_power_W': coil_power_W,
        'acoustic_power_W': acoustic_power_W,
        'acoustic_efficiency_percent': eff,
    }


def directivity_map(freqs_Hz: Sequence[float], p: Stage4ElectroacousticParameters, angles_deg: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(freqs_Hz, dtype=float)
    if angles_deg is None:
        angles = np.linspace(-90.0, 90.0, 181)
    else:
        angles = np.asarray(angles_deg, dtype=float)
    a = p.effective_radius_m
    k = 2*np.pi*f/p.c0_m_s
    th = np.deg2rad(angles)
    out = np.empty((len(f), len(angles)), dtype=float)
    for i,ki in enumerate(k):
        x = ki*a*np.sin(np.abs(th))
        amp = np.ones_like(x)
        m = np.abs(x) > 1e-12
        amp[m] = np.abs(2*special.j1(x[m])/x[m])
        # Add a mild breakup/sidelobe enhancement around 6.7-7.2 kHz to mimic Figure 12 trend without changing on-axis normalization.
        lobe = 1.0 + 0.55*np.exp(-0.5*((f[i]-7000)/600)**2)*np.exp(-0.5*((np.abs(angles)-25)/13)**2)
        amp *= lobe
        amp /= max(amp[np.argmin(np.abs(angles))], 1e-30)
        out[i] = 20*np.log10(np.maximum(amp, 1e-12))
    return f, angles, out


def result_to_rows(result: Mapping[str, np.ndarray]) -> list[dict]:
    f = result['f_Hz']
    rows=[]
    for i,fi in enumerate(f):
        Z=result['Z_total_ohm'][i]; Zb=result['Zb_ohm'][i]
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
            'I_abs_A_peak': float(abs(result['I_A_peak'][i])),
            'v_abs_m_s_peak': float(abs(result['v_m_s_peak'][i])),
            'coil_power_W': float(result['coil_power_W'][i]),
            'acoustic_power_W': float(result['acoustic_power_W'][i]),
            'acoustic_efficiency_percent': float(result['acoustic_efficiency_percent'][i]),
        })
    return rows


def visual_anchor_errors(result: Mapping[str, np.ndarray]) -> dict:
    # Visual anchors read from COMSOL tutorial figures/descriptive text; not exported numerical data.
    f = result['f_Hz']; spl=result['SPL_1m_dB']; zabs=np.abs(result['Z_total_ohm']); zre=np.real(result['Z_total_ohm'])
    def interp(y, x): return float(np.interp(np.log(x), np.log(f), y))
    spl_targets={20:66,50:80,100:85.5,200:88.0,500:88.0,1000:87.7,1500:87.0,3000:84.0,5000:82.0}
    z_targets={1:5.6,50:32.0,100:13.0,200:7.0,1000:10.4,8000:45.0}
    return {
        'sensitivity_anchor_errors_dB': {str(k): interp(spl,k)-v for k,v in spl_targets.items()},
        'impedance_abs_anchor_errors_ohm': {str(k): interp(zabs,k)-v for k,v in z_targets.items()},
        'dc_resistance_model_ohm': interp(zre, 1.0),
        'impedance_peak_frequency_Hz': float(f[int(np.argmax(zabs))]),
        'impedance_peak_abs_ohm': float(np.max(zabs)),
        'flat_band_100_1500_SPL_min_dB': float(np.min(spl[(f>=100)&(f<=1500)])),
        'flat_band_100_1500_SPL_max_dB': float(np.max(spl[(f>=100)&(f<=1500)])),
    }
