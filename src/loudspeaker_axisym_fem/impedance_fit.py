from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class TSFitParams:
    Re: float
    Le: float
    n_e: float
    Rms: float
    Mms: float
    Cms: float
    BL: float
    Sd: float = math.pi * 0.12**2
    rho0: float = 1.2
    c0: float = 343.0

    @property
    def Fs(self) -> float:
        return 1.0 / (2.0 * math.pi * math.sqrt(max(self.Mms * self.Cms, 1e-300)))

    @property
    def Qms(self) -> float:
        return 2.0 * math.pi * self.Fs * self.Mms / max(self.Rms, 1e-300)

    @property
    def Qes(self) -> float:
        return 2.0 * math.pi * self.Fs * self.Mms * self.Re / max(self.BL**2, 1e-300)

    @property
    def Qts(self) -> float:
        return self.Qms * self.Qes / max(self.Qms + self.Qes, 1e-300)

    @property
    def Vas(self) -> float:
        return self.rho0 * self.c0**2 * self.Sd**2 * self.Cms

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.update(Fs=self.Fs, Qms=self.Qms, Qes=self.Qes, Qts=self.Qts,
                 Vas_m3=self.Vas, Vas_liter=1000.0*self.Vas)
        return {k: float(v) for k, v in d.items()}


def identifiable_composites(pars: TSFitParams) -> dict[str, float]:
    """Combinations identifiable from a single free-air complex impedance curve.

    A single free-air impedance curve identifies the electrical branch and the
    motional impedance shape, but not Mms/Cms/Rms/BL separately unless one of
    the mechanical scale parameters is constrained by added-mass, known Mms, or
    independent force-factor information.
    """
    return {
        'Fs': pars.Fs,
        'BL2_over_Rms': pars.BL**2 / max(pars.Rms, 1e-300),
        'BL2_over_Mms': pars.BL**2 / max(pars.Mms, 1e-300),
        'BL2_times_Cms': pars.BL**2 * pars.Cms,
        'Qms': pars.Qms,
        'Qes': pars.Qes,
        'Qts': pars.Qts,
    }


def fractional_coil_impedance(f: np.ndarray, Re: float, Le: float, n_e: float, Rg: float = 0.0) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    w = 2.0 * np.pi * np.maximum(f, 1e-300)
    n_e = np.clip(n_e, 0.05, 0.95)
    L_E = (Le / np.sin(n_e * np.pi / 2.0)) * w ** (n_e - 1.0)
    Rp_E = (Le / np.cos(n_e * np.pi / 2.0)) * w ** n_e
    Zpar = 1.0 / (1.0 / (1j * w * L_E) + 1.0 / Rp_E)
    return Rg + Re + Zpar


def free_air_impedance(f: np.ndarray, pars: TSFitParams) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    w = 2.0 * np.pi * np.maximum(f, 1e-300)
    Ze = fractional_coil_impedance(f, pars.Re, pars.Le, pars.n_e)
    Zm = pars.Rms + 1j * w * pars.Mms + 1.0 / (1j * w * pars.Cms)
    return Ze + pars.BL**2 / Zm


_PARAM_NAMES = ('Re', 'Le', 'n_e', 'Rms', 'Mms', 'Cms', 'BL')
_LB = dict(Re=0.1, Le=1e-6, n_e=0.2, Rms=0.05, Mms=1e-3, Cms=1e-5, BL=0.2)
_UB = dict(Re=30.0, Le=0.2, n_e=0.95, Rms=50.0, Mms=0.5, Cms=0.05, BL=60.0)


def _pack(p: TSFitParams) -> np.ndarray:
    return np.array([getattr(p, k) for k in _PARAM_NAMES], dtype=float)


def _unpack_full(values: Mapping[str, float], Sd: float, rho0: float, c0: float) -> TSFitParams:
    return TSFitParams(Re=float(values['Re']), Le=float(values['Le']), n_e=float(values['n_e']),
                       Rms=float(values['Rms']), Mms=float(values['Mms']), Cms=float(values['Cms']),
                       BL=float(values['BL']), Sd=Sd, rho0=rho0, c0=c0)


def _normalise_fixed(fixed: Mapping[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (fixed or {}).items():
        if v is None:
            continue
        if k not in _PARAM_NAMES:
            raise ValueError(f'Unsupported fixed parameter {k!r}; use one of {_PARAM_NAMES}.')
        vv = float(v)
        if not (_LB[k] <= vv <= _UB[k]):
            raise ValueError(f'Fixed {k}={vv:g} is outside [{_LB[k]:g}, {_UB[k]:g}].')
        out[k] = vv
    return out


def _variable_vectors(initial: TSFitParams, fixed: Mapping[str, float]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = [k for k in _PARAM_NAMES if k not in fixed]
    x0 = np.array([getattr(initial, k) for k in names], dtype=float)
    lb = np.array([_LB[k] for k in names], dtype=float)
    ub = np.array([_UB[k] for k in names], dtype=float)
    if x0.size:
        x0 = np.minimum(np.maximum(x0, lb*1.0001), ub*0.9999)
    return names, x0, lb, ub


def _params_from_variable(x: np.ndarray, names: list[str], initial: TSFitParams, fixed: Mapping[str, float], Sd: float, rho0: float, c0: float) -> TSFitParams:
    vals = {k: getattr(initial, k) for k in _PARAM_NAMES}
    vals.update({k: float(v) for k, v in fixed.items()})
    vals.update({k: float(v) for k, v in zip(names, x)})
    return _unpack_full(vals, Sd=Sd, rho0=rho0, c0=c0)


def initial_guess_from_curve(f: np.ndarray, z: np.ndarray, Sd: float, rho0: float, c0: float) -> TSFitParams:
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=complex)
    za = np.abs(z)
    low = f <= min(80.0, np.nanmax(f))
    Re0 = float(np.nanpercentile(np.real(z[low]) if np.any(low) else za, 10))
    Re0 = max(Re0, 0.2)
    high = f >= np.nanpercentile(f, 75)
    if np.any(high):
        Le0 = float(np.nanmedian(np.maximum(np.abs(z[high] - Re0), 1e-6) / (2*np.pi*np.maximum(f[high], 1.0))**0.7))
    else:
        Le0 = 3e-3
    Le0 = min(max(Le0, 1e-5), 0.05)
    ip = int(np.nanargmax(za))
    Fs0 = float(f[ip])
    Fs0 = min(max(Fs0, 10.0), 500.0)
    Mms0 = 0.03
    Cms0 = 1.0 / ((2*np.pi*Fs0)**2 * Mms0)
    Zmax = max(float(za[ip]), Re0*1.2)
    Rms0 = 2.0
    BL0 = math.sqrt(max((Zmax - Re0) * Rms0, 1e-6))
    return TSFitParams(Re=Re0, Le=Le0, n_e=0.7, Rms=Rms0, Mms=Mms0, Cms=Cms0, BL=BL0,
                       Sd=Sd, rho0=rho0, c0=c0)


def fit_free_air_impedance(
    f: np.ndarray,
    z_meas: np.ndarray,
    initial: TSFitParams | None = None,
    fit_mode: str = 'complex',
    Sd: float = math.pi * 0.12**2,
    rho0: float = 1.2,
    c0: float = 343.0,
    max_nfev: int = 6000,
    fixed: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit free-air impedance to a T/S + fractional voice-coil model.

    Use `fixed={'Mms': value}` or another mechanical-scale constraint when the
    objective is absolute T/S identification. Without a mechanical constraint,
    only the motional composite terms are unique; the code still fits a valid
    impedance-equivalent model and reports this limitation explicitly.
    """
    f = np.asarray(f, dtype=float)
    z_meas = np.asarray(z_meas, dtype=complex)
    valid = np.isfinite(f) & np.isfinite(np.real(z_meas)) & np.isfinite(np.imag(z_meas)) & (f > 0)
    f, z_meas = f[valid], z_meas[valid]
    order = np.argsort(f)
    f, z_meas = f[order], z_meas[order]
    if f.size < 8:
        raise ValueError('At least eight frequency points are recommended for T/S fitting.')
    fixed_n = _normalise_fixed(fixed)
    if initial is None:
        initial = initial_guess_from_curve(f, z_meas, Sd=Sd, rho0=rho0, c0=c0)
    # If the user supplied a fixed scale parameter, seed the dependent variables
    # close to the impedance-equivalent initial shape but with the fixed value.
    init_vals = {k: getattr(initial, k) for k in _PARAM_NAMES}
    init_vals.update(fixed_n)
    initial = _unpack_full(init_vals, Sd=Sd, rho0=rho0, c0=c0)
    names, x0, lb, ub = _variable_vectors(initial, fixed_n)

    complex_scale = np.maximum(np.nanmedian(np.abs(z_meas)), 1.0)

    def residual(x: np.ndarray) -> np.ndarray:
        pars = _params_from_variable(x, names, initial, fixed_n, Sd=Sd, rho0=rho0, c0=c0)
        z_pred = free_air_impedance(f, pars)
        if fit_mode == 'magnitude':
            return (20.0*np.log10(np.maximum(np.abs(z_pred), 1e-12)) -
                    20.0*np.log10(np.maximum(np.abs(z_meas), 1e-12))) / 0.25
        if fit_mode == 'complex':
            r = (z_pred - z_meas) / complex_scale
            return np.concatenate([np.real(r), np.imag(r)])
        raise ValueError(f'unsupported fit_mode={fit_mode!r}')

    if len(names):
        opt = least_squares(residual, x0, bounds=(lb, ub), x_scale='jac', loss='soft_l1', f_scale=1.0, max_nfev=max_nfev)
        pars = _params_from_variable(opt.x, names, initial, fixed_n, Sd=Sd, rho0=rho0, c0=c0)
        success = bool(opt.success); status = int(opt.status); message = str(opt.message); cost = float(opt.cost); nfev = int(opt.nfev)
    else:
        pars = initial
        success = True; status = 0; message = 'all parameters fixed; no optimization run'; cost = float(np.sum(residual(np.array([]))**2)); nfev = 0
    z_fit = free_air_impedance(f, pars)
    err = z_fit - z_meas
    rms_complex_ohm = float(np.sqrt(np.nanmean(np.abs(err)**2)))
    rms_mag_db = float(np.sqrt(np.nanmean((20*np.log10(np.maximum(np.abs(z_fit), 1e-12)) - 20*np.log10(np.maximum(np.abs(z_meas), 1e-12)))**2)))
    unconstrained_mechanics = len({'Rms','Mms','Cms','BL'} & set(names)) >= 4 and not ({'Rms','Mms','Cms','BL'} & set(fixed_n))
    return dict(
        params=pars,
        params_dict=pars.to_dict(),
        identifiable_composites=identifiable_composites(pars),
        f=f,
        z_fit=z_fit,
        z_meas=z_meas,
        success=success,
        status=status,
        message=message,
        cost=cost,
        nfev=nfev,
        rms_complex_ohm=rms_complex_ohm,
        rms_mag_db=rms_mag_db,
        fit_mode=fit_mode,
        fixed=dict(fixed_n),
        fitted_parameters=names,
        underidentified=bool(unconstrained_mechanics),
    )


def read_impedance_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with open(path, newline='', encoding='utf-8-sig') as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f'empty CSV: {path}')
    cols = {c.lower().strip(): c for c in rows[0].keys()}
    def find(*names: str) -> str | None:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        for k, v in cols.items():
            if any(n.lower() in k for n in names):
                return v
        return None
    cf = find('f_hz', 'frequency', 'freq', 'f')
    cre = find('z_re_ohm', 're_ohm', 'z_re', 'real', 're')
    cim = find('z_im_ohm', 'im_ohm', 'z_im', 'imag', 'im')
    cab = find('z_abs_ohm', 'abs_ohm', '|z|', 'magnitude', 'z_abs')
    if cf is None:
        raise ValueError('CSV must contain a frequency column, e.g. f_Hz.')
    f = np.array([float(r[cf]) for r in rows], dtype=float)
    if cre is not None and cim is not None:
        z = np.array([float(r[cre]) + 1j*float(r[cim]) for r in rows], dtype=complex)
    elif cab is not None:
        z = np.array([float(r[cab]) + 0j for r in rows], dtype=complex)
    else:
        raise ValueError('CSV must contain either real+imag columns or an impedance magnitude column.')
    return f, z


def write_fit_report(outdir: str | Path, fit: Mapping[str, Any]) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'impedance_fit_params.json').write_text(json.dumps(fit['params_dict'], indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / 'impedance_fit_composites.json').write_text(json.dumps(fit['identifiable_composites'], indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 实测阻抗 T/S 拟合报告', '', f"- 拟合模式：`{fit['fit_mode']}`", f"- 收敛：{fit['success']}，迭代次数：{fit['nfev']}", f"- 复阻抗 RMS 误差：{fit['rms_complex_ohm']:.6g} Ω", f"- 幅值 RMS 误差：{fit['rms_mag_db']:.6g} dB", f"- 固定参数：{fit.get('fixed', {})}", '']
    if fit.get('underidentified'):
        lines += ['**识别性提示：** 单条自由场阻抗曲线不能唯一确定 `Rms/Mms/Cms/BL` 的绝对尺度；它只能唯一确定 motional composite。请固定 Mms/BL，或使用 added-mass / sealed-box 双曲线数据。', '']
    lines += ['## 识别参数', '']
    for k, v in fit['params_dict'].items():
        unit = ''
        if k == 'Le': unit = ' H'
        elif k in ('Mms',): unit = ' kg'
        elif k in ('Cms',): unit = ' m/N'
        elif k in ('Fs',): unit = ' Hz'
        elif k in ('Vas_m3',): unit = ' m^3'
        elif k in ('Vas_liter',): unit = ' L'
        elif k in ('Re',): unit = ' Ω'
        lines.append(f'- {k}: {v:.8g}{unit}')
    lines += ['', '## 单曲线可唯一识别的组合量', '']
    for k, v in fit['identifiable_composites'].items():
        lines.append(f'- {k}: {v:.8g}')
    if fit['fit_mode'] == 'magnitude':
        lines += ['', '注意：幅值-only 拟合不能唯一确定全部 T/S 参数，应作为初值或质控工具；正式识别建议输入复阻抗。']
    (outdir / 'impedance_fit_report.md').write_text('\n'.join(lines), encoding='utf-8')
