from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import json
import math

import numpy as np

try:
    from .fem_solver import LoudspeakerParams
except Exception:  # pragma: no cover
    LoudspeakerParams = None  # type: ignore


def _arr(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _finite_xy(f: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    ff = _arr(f, float)
    yy = _arr(y, float)
    m = np.isfinite(ff) & np.isfinite(yy)
    ff, yy = ff[m], yy[m]
    order = np.argsort(ff)
    return ff[order], yy[order]


def interp_logx(f: Iterable[float], y: Iterable[float], x: float) -> float:
    """Interpolate y at frequency x using log-frequency abscissa.

    Frequency-response engineering metrics are normally read on a logarithmic
    axis; this helper avoids bias from sparse ISO-band sampling.
    """
    ff, yy = _finite_xy(f, y)
    if ff.size == 0:
        return float('nan')
    x = float(x)
    if x <= 0:
        return float(np.interp(x, ff, yy))
    return float(np.interp(np.log(x), np.log(np.maximum(ff, 1e-300)), yy))


def _crossing_logx(f: np.ndarray, y: np.ndarray, level: float, start: int, direction: int) -> float:
    """Return log-frequency crossing of y with level from start index.

    direction=-1 searches downward in frequency; direction=+1 searches upward.
    """
    if direction < 0:
        rng = range(start, 0, -1)
        for i in rng:
            y0, y1 = y[i - 1], y[i]
            if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
                x0, x1 = np.log(f[i - 1]), np.log(f[i])
                t = (level - y0) / (y1 - y0)
                return float(np.exp(x0 + t * (x1 - x0)))
    else:
        rng = range(start, len(f) - 1)
        for i in rng:
            y0, y1 = y[i], y[i + 1]
            if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
                x0, x1 = np.log(f[i]), np.log(f[i + 1])
                t = (level - y0) / (y1 - y0)
                return float(np.exp(x0 + t * (x1 - x0)))
    return float('nan')


@dataclass(frozen=True)
class ImpedancePeakMetrics:
    peak_frequency_hz: float
    peak_abs_ohm: float
    peak_re_ohm: float
    peak_im_ohm: float
    search_min_hz: float
    search_max_hz: float
    threshold_ohm: float
    f1_hz: float
    f2_hz: float
    qms_est: float
    qes_est: float
    qts_est: float
    peak_ratio_to_re: float


@dataclass(frozen=True)
class EngineeringMetrics:
    case: str
    n_frequencies: int
    f_min_hz: float
    f_max_hz: float
    fs_parameter_hz: float
    qts_parameter: float
    qms_parameter: float
    qes_parameter: float
    vas_liter: float
    eta0_percent: float
    re_low_ohm: float
    z_peak_full_freq_hz: float
    z_peak_full_abs_ohm: float
    z_peak_below_500_freq_hz: float
    z_peak_below_500_abs_ohm: float
    z_peak_below_500_qms_est: float
    z_peak_below_500_qes_est: float
    z_peak_below_500_qts_est: float
    f3_low_hz: float
    passband_ref_spl_db: float
    spl_100_hz_db: float
    spl_200_hz_db: float
    spl_700_hz_db: float
    spl_1000_hz_db: float
    spl_283v_1000_hz_db: float
    u_200_hz_m_s: float
    u_1000_hz_m_s: float
    f_back_200_hz_rms_n: float
    f_back_peak_below_500_hz: float
    f_back_peak_below_500_rms_n: float
    p_e_1000_hz_w: float
    eta_1000_hz_percent: float
    notes: str


def impedance_peak_metrics(
    f: Iterable[float],
    z_complex: Iterable[complex],
    re_reference: float | None = None,
    search_min_hz: float = 10.0,
    search_max_hz: float = 500.0,
) -> ImpedancePeakMetrics:
    ff = np.asarray(f, dtype=float)
    zz = np.asarray(z_complex, dtype=complex)
    mask = np.isfinite(ff) & np.isfinite(np.real(zz)) & np.isfinite(np.imag(zz)) & (ff >= search_min_hz) & (ff <= search_max_hz)
    if not np.any(mask):
        nan = float('nan')
        return ImpedancePeakMetrics(nan, nan, nan, nan, search_min_hz, search_max_hz, nan, nan, nan, nan, nan, nan, nan)
    f2 = ff[mask]
    z2 = zz[mask]
    order = np.argsort(f2)
    f2, z2 = f2[order], z2[order]
    za = np.abs(z2)
    ip = int(np.nanargmax(za))
    fp = float(f2[ip])
    zp = float(za[ip])
    if re_reference is None or not np.isfinite(re_reference) or re_reference <= 0:
        lowmask = f2 <= min(80.0, search_max_hz)
        re_reference = float(np.nanmedian(np.real(z2[lowmask]))) if np.any(lowmask) else float(np.nanmin(za))
    re_reference = max(float(re_reference), 1e-9)
    ratio = zp / re_reference
    threshold = math.sqrt(max(zp * re_reference, 1e-18))
    f1 = _crossing_logx(f2, za, threshold, ip, -1)
    f3 = _crossing_logx(f2, za, threshold, ip, +1)
    if np.isfinite(f1) and np.isfinite(f3) and f3 > f1 and ratio > 1:
        # Standard loudspeaker impedance-curve estimate using f1/f2 at sqrt(Re*Zmax).
        qms = fp * math.sqrt(ratio) / (f3 - f1)
        qes = qms / max(ratio - 1.0, 1e-12)
        qts = qms / max(ratio, 1e-12)
    else:
        qms = qes = qts = float('nan')
    return ImpedancePeakMetrics(
        peak_frequency_hz=fp,
        peak_abs_ohm=zp,
        peak_re_ohm=float(np.real(z2[ip])),
        peak_im_ohm=float(np.imag(z2[ip])),
        search_min_hz=float(search_min_hz),
        search_max_hz=float(search_max_hz),
        threshold_ohm=float(threshold),
        f1_hz=float(f1),
        f2_hz=float(f3),
        qms_est=float(qms),
        qes_est=float(qes),
        qts_est=float(qts),
        peak_ratio_to_re=float(ratio),
    )


def estimate_low_f3(f: Iterable[float], spl_db: Iterable[float], ref_band: tuple[float, float] = (500.0, 1000.0)) -> tuple[float, float]:
    ff, yy = _finite_xy(f, spl_db)
    if ff.size < 3:
        return float('nan'), float('nan')
    band = (ff >= ref_band[0]) & (ff <= ref_band[1])
    if np.any(band):
        ref = float(np.nanmean(yy[band]))
    else:
        ref = interp_logx(ff, yy, math.sqrt(ref_band[0] * ref_band[1]))
    target = ref - 3.0
    below = ff < ref_band[0]
    idxs = np.nonzero(below)[0]
    if idxs.size < 2:
        return float('nan'), ref
    # find the highest-frequency crossing on the low side.
    for i in idxs[::-1]:
        if i <= 0:
            continue
        if (yy[i - 1] - target) * (yy[i] - target) <= 0 and yy[i - 1] != yy[i]:
            x0, x1 = np.log(ff[i - 1]), np.log(ff[i])
            t = (target - yy[i - 1]) / (yy[i] - yy[i - 1])
            return float(np.exp(x0 + t * (x1 - x0))), ref
    return float('nan'), ref


def extract_case_metrics(case: str, result: Mapping[str, Any], params: Any | None = None, peak_max_hz: float = 500.0) -> EngineeringMetrics:
    f = np.asarray(result['f'], dtype=float)
    z = np.asarray(result['Zvoice'], dtype=complex)
    spl_key = 'SPL_1m_HK' if 'SPL_1m_HK' in result else 'SPL_1m'
    spl = np.asarray(result[spl_key], dtype=float)
    u = np.asarray(result['u_D'], dtype=complex)
    fback = np.asarray(result.get('F_back_rms', np.full_like(f, np.nan)), dtype=float)
    pe = np.asarray(result.get('P_E', np.full_like(f, np.nan)), dtype=float)
    eta = np.asarray(result.get('eta', np.full_like(f, np.nan)), dtype=float)

    low_for_re = (f >= max(np.nanmin(f), 10.0)) & (f <= min(80.0, peak_max_hz))
    re_low = float(np.nanmedian(np.real(z[low_for_re]))) if np.any(low_for_re) else float(np.nanmin(np.abs(z)))
    pk_low = impedance_peak_metrics(f, z, re_reference=re_low, search_min_hz=max(10.0, float(np.nanmin(f))), search_max_hz=peak_max_hz)
    iz_full = int(np.nanargmax(np.abs(z)))
    f3, refspl = estimate_low_f3(f, spl)
    low_mask = (f <= peak_max_hz) & np.isfinite(fback)
    if np.any(low_mask):
        idxs = np.nonzero(low_mask)[0]
        ib = idxs[int(np.nanargmax(fback[low_mask]))]
        fback_pk_f = float(f[ib])
        fback_pk_v = float(fback[ib])
    else:
        fback_pk_f = fback_pk_v = float('nan')

    fs = qts = qms = qes = vas_l = eta0 = float('nan')
    if params is not None:
        fs = float(getattr(params, 'Fs', float('nan')))
        qts = float(getattr(params, 'Q_TS', float('nan')))
        qms = float(getattr(params, 'Q_MS', float('nan')))
        qes = float(getattr(params, 'Q_ES', float('nan')))
        vas_l = float(getattr(params, 'V_AS', float('nan')) * 1000.0)
        eta0 = float(getattr(params, 'eta0', float('nan')) * 100.0)

    notes = []
    if float(f[iz_full]) > peak_max_hz:
        notes.append('full-band impedance maximum is outside the low-frequency resonance window; use z_peak_below_500 for box/driver resonance.')
    if not np.isfinite(pk_low.qts_est):
        notes.append('Q estimate unavailable: impedance curve did not cross sqrt(Re*Zmax) on both sides in the available frequency grid.')
    if not np.isfinite(f3):
        notes.append('F3 unavailable: no -3 dB low-side crossing relative to the selected passband reference.')

    return EngineeringMetrics(
        case=case,
        n_frequencies=int(len(f)),
        f_min_hz=float(np.nanmin(f)),
        f_max_hz=float(np.nanmax(f)),
        fs_parameter_hz=fs,
        qts_parameter=qts,
        qms_parameter=qms,
        qes_parameter=qes,
        vas_liter=vas_l,
        eta0_percent=eta0,
        re_low_ohm=re_low,
        z_peak_full_freq_hz=float(f[iz_full]),
        z_peak_full_abs_ohm=float(np.abs(z[iz_full])),
        z_peak_below_500_freq_hz=pk_low.peak_frequency_hz,
        z_peak_below_500_abs_ohm=pk_low.peak_abs_ohm,
        z_peak_below_500_qms_est=pk_low.qms_est,
        z_peak_below_500_qes_est=pk_low.qes_est,
        z_peak_below_500_qts_est=pk_low.qts_est,
        f3_low_hz=f3,
        passband_ref_spl_db=refspl,
        spl_100_hz_db=interp_logx(f, spl, 100.0),
        spl_200_hz_db=interp_logx(f, spl, 200.0),
        spl_700_hz_db=interp_logx(f, spl, 700.0),
        spl_1000_hz_db=interp_logx(f, spl, 1000.0),
        spl_283v_1000_hz_db=interp_logx(f, spl, 1000.0) + 20.0 * math.log10(2.83),
        u_200_hz_m_s=interp_logx(f, np.abs(u), 200.0),
        u_1000_hz_m_s=interp_logx(f, np.abs(u), 1000.0),
        f_back_200_hz_rms_n=interp_logx(f, fback, 200.0),
        f_back_peak_below_500_hz=fback_pk_f,
        f_back_peak_below_500_rms_n=fback_pk_v,
        p_e_1000_hz_w=interp_logx(f, pe, 1000.0),
        eta_1000_hz_percent=interp_logx(f, 100.0 * eta, 1000.0),
        notes='; '.join(notes),
    )


def metrics_to_dict(m: EngineeringMetrics) -> Dict[str, Any]:
    return asdict(m)


def write_metrics_outputs(metrics: list[EngineeringMetrics], outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [metrics_to_dict(m) for m in metrics]
    (outdir / 'engineering_metrics_summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    if rows:
        import csv
        with open(outdir / 'engineering_metrics_summary.csv', 'w', newline='', encoding='utf-8') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    lines = ['# 扬声器工程指标汇总', '', '这些指标用于替代单纯全频最大值，避免把高频音圈电感上升误判为低频箱体/单元共振。', '']
    for m in metrics:
        lines.append(f"## {m.case}")
        lines.append('')
        lines.append(f"- 低频阻抗峰：{m.z_peak_below_500_abs_ohm:.3g} Ω @ {m.z_peak_below_500_freq_hz:.3g} Hz")
        lines.append(f"- 估算 Q：Qms={m.z_peak_below_500_qms_est:.3g}, Qes={m.z_peak_below_500_qes_est:.3g}, Qts/Qtc={m.z_peak_below_500_qts_est:.3g}")
        lines.append(f"- SPL：200 Hz={m.spl_200_hz_db:.2f} dB, 1 kHz={m.spl_1000_hz_db:.2f} dB @ 1 Vrms, 1 kHz={m.spl_283v_1000_hz_db:.2f} dB @ 2.83 Vrms")
        lines.append(f"- 后腔力：200 Hz={m.f_back_200_hz_rms_n:.3g} N, 低频峰={m.f_back_peak_below_500_rms_n:.3g} N @ {m.f_back_peak_below_500_hz:.3g} Hz")
        if m.notes:
            lines.append(f"- 注意：{m.notes}")
        lines.append('')
    (outdir / 'engineering_metrics_summary.md').write_text('\n'.join(lines), encoding='utf-8')
