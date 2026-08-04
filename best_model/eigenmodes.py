from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import linear_sum_assignment

@dataclass
class EigenResult:
    frequencies_Hz: np.ndarray
    eigenvalues_rad2_s2: np.ndarray
    vectors_free: np.ndarray
    free_dofs: np.ndarray


def solve_p2_eigenmodes(solid, n_modes: int = 40, *, sigma_Hz: float = 0.0, tol: float = 1e-10, maxiter: int = 8000) -> EigenResult:
    free = np.asarray(solid.free_dofs, int)
    K = solid.K_real[free][:, free].tocsc()
    M = solid.M[free][:, free].tocsc()
    sigma = (2.0 * math.pi * float(sigma_Hz)) ** 2
    try:
        vals, vecs = eigsh(K, k=int(n_modes), M=M, sigma=sigma, which="LM", tol=float(tol), maxiter=int(maxiter))
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            raise
        vals, vecs = exc.eigenvalues, exc.eigenvectors
    order = np.argsort(np.real(vals))
    vals = np.real(vals[order])
    vecs = vecs[:, order]
    freqs = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * math.pi)
    return EigenResult(freqs, vals, vecs, free)


def full_vectors(result: EigenResult, ndof: int) -> np.ndarray:
    out = np.zeros((ndof, result.vectors_free.shape[1]), complex)
    out[result.free_dofs, :] = result.vectors_free
    return out


def _complex_columns(df: pd.DataFrame, real_name: str, imag_name: str) -> np.ndarray:
    return df[real_name].to_numpy(float) + 1j * df[imag_name].to_numpy(float)


def interpolate_comsol_modes_to_p2_nodes(solid, shapes_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate exported COMSOL (u,w) modes to the P2 structural nodes.

    Linear interpolation is used inside the sampled structural cloud and nearest
    interpolation only for points just outside a convex hull due to curved-edge
    sampling. Duplicate coordinates at material interfaces are averaged first.
    """
    df = pd.read_csv(shapes_csv)
    fcol = "freq_Hz"
    freqs = np.array(sorted(df[fcol].dropna().unique()), float)
    xy_target = np.asarray(solid.points_rz_m, float)
    modes = np.zeros((solid.ndof, len(freqs)), complex)
    for j, f in enumerate(freqs):
        q = df[np.isclose(df[fcol].astype(float), f)].copy()
        rcol = "r/1[m]_real" if "r/1[m]_real" in q.columns else "r_m"
        zcol = "z/1[m]_real" if "z/1[m]_real" in q.columns else "z_m"
        q["_ur"] = _complex_columns(q, "u_real", "u_imag")
        q["_uz"] = _complex_columns(q, "w_real", "w_imag")
        # Continuity makes averaging duplicated interface nodes appropriate.
        g = q.groupby([rcol, zcol], as_index=False).agg({"_ur": "mean", "_uz": "mean"})
        xy = g[[rcol, zcol]].to_numpy(float)
        urv = g["_ur"].to_numpy(complex)
        uzv = g["_uz"].to_numpy(complex)
        def interp(values):
            lin_r = LinearNDInterpolator(xy, values.real, fill_value=np.nan)
            lin_i = LinearNDInterpolator(xy, values.imag, fill_value=np.nan)
            out = np.asarray(lin_r(xy_target)) + 1j * np.asarray(lin_i(xy_target))
            bad = ~np.isfinite(out.real) | ~np.isfinite(out.imag)
            if np.any(bad):
                nr = NearestNDInterpolator(xy, values.real)
                ni = NearestNDInterpolator(xy, values.imag)
                out[bad] = nr(xy_target[bad]) + 1j * ni(xy_target[bad])
            return out
        ur = interp(urv); uz = interp(uzv)
        modes[0::2, j] = ur
        modes[1::2, j] = uz
    return freqs, modes


def mass_mac_matrix(solid, python_modes_full: np.ndarray, comsol_modes_full: np.ndarray) -> np.ndarray:
    M = solid.M.astype(complex)
    Mp = M @ python_modes_full
    Mc = M @ comsol_modes_full
    npow = np.real(np.sum(np.conj(python_modes_full) * Mp, axis=0))
    cpow = np.real(np.sum(np.conj(comsol_modes_full) * Mc, axis=0))
    cross = np.conj(python_modes_full).T @ Mc
    den = np.maximum(npow[:, None] * cpow[None, :], 1e-300)
    return np.clip(np.abs(cross) ** 2 / den, 0.0, 1.0)


def pair_modes(python_freqs: np.ndarray, comsol_freqs: np.ndarray, mac: np.ndarray, *, frequency_weight: float = 0.08) -> list[dict]:
    # rows Python, columns COMSOL; pair every COMSOL mode uniquely.
    flog = np.abs(np.log(np.maximum(python_freqs[:, None], 1e-12) / np.maximum(comsol_freqs[None, :], 1e-12)))
    cost = (1.0 - mac) + float(frequency_weight) * flog
    prow, ccol = linear_sum_assignment(cost)
    pairs=[]
    for p,c in zip(prow,ccol):
        pairs.append({
            "COMSOL_mode": int(c+1), "Python_mode": int(p+1),
            "COMSOL_Hz": float(comsol_freqs[c]), "Python_Hz": float(python_freqs[p]),
            "frequency_error_percent": float(100.0*(python_freqs[p]/comsol_freqs[c]-1.0)),
            "mass_MAC": float(mac[p,c]), "assignment_cost": float(cost[p,c]),
        })
    return sorted(pairs,key=lambda x:x["COMSOL_mode"])


def _frequency_only_pairing(python_freqs: np.ndarray, comsol_freqs: np.ndarray) -> list[dict]:
    cost=np.abs(np.log(np.maximum(python_freqs[:,None],1e-12)/np.maximum(comsol_freqs[None,:],1e-12)))
    pr,cr=linear_sum_assignment(cost)
    rows=[]
    for p,c in zip(pr,cr):
        rows.append({"COMSOL_mode":int(c+1),"Python_mode":int(p+1),"COMSOL_Hz":float(comsol_freqs[c]),"Python_Hz":float(python_freqs[p]),"frequency_error_percent":float(100*(python_freqs[p]/comsol_freqs[c]-1)),"pairing_basis":"frequency_only_NO_MAC"})
    return sorted(rows,key=lambda x:x['COMSOL_mode'])


def write_eigen_outputs(outdir: str | Path, solid, result: EigenResult, *, comsol_shapes_csv: str | Path | None = None, comsol_frequencies_csv: str | Path | None = None) -> dict:
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    pyfull=full_vectors(result,solid.ndof)
    np.savez_compressed(out/"p2_eigenmodes.npz",frequencies_Hz=result.frequencies_Hz,eigenvalues_rad2_s2=result.eigenvalues_rad2_s2,vectors_full=pyfull,points_rz_m=solid.points_rz_m,triangles6=solid.triangles6,domains=solid.domains,free_dofs=result.free_dofs)
    pd.DataFrame({"Python_mode":np.arange(1,len(result.frequencies_Hz)+1),"frequency_Hz":result.frequencies_Hz}).to_csv(out/"p2_eigenfrequencies.csv",index=False)
    summary={"n_modes":int(len(result.frequencies_Hz)),"python_frequencies_Hz":result.frequencies_Hz.tolist()}
    cf_all=None
    if comsol_frequencies_csv:
        fd=pd.read_csv(comsol_frequencies_csv);cf_all=fd["eigenfrequency_real_Hz"].to_numpy(float)
        fpairs=_frequency_only_pairing(result.frequencies_Hz,cf_all)
        pd.DataFrame(fpairs).to_csv(out/"frequency_only_pairing.csv",index=False)
        summary.update({"COMSOL_frequencies_Hz":cf_all.tolist(),"frequency_only_pairs":fpairs})
    if comsol_shapes_csv:
        cf_shape,cmodes=interpolate_comsol_modes_to_p2_nodes(solid,comsol_shapes_csv)
        mac=mass_mac_matrix(solid,pyfull,cmodes)
        pairs=pair_modes(result.frequencies_Hz,cf_shape,mac)
        pd.DataFrame(mac,index=np.arange(1,len(result.frequencies_Hz)+1),columns=[f"shape_{i+1}_{f:.6g}Hz" for i,f in enumerate(cf_shape)]).to_csv(out/"mass_MAC_matrix.csv",index_label="Python_mode")
        pd.DataFrame(pairs).to_csv(out/"mode_shape_MAC_pairing.csv",index=False)
        missing=0 if cf_all is None else max(len(cf_all)-len(cf_shape),0)
        status="COMPLETE" if missing==0 else "PARTIAL_BLOCKED_MISSING_COMSOL_MODE_SHAPES"
        summary.update({"COMSOL_shape_frequencies_Hz":cf_shape.tolist(),"n_COMSOL_shapes":int(len(cf_shape)),"mode_shape_pairs":pairs,"mean_paired_MAC":float(np.mean([x['mass_MAC'] for x in pairs])) if pairs else None,"MAC_pairing_status":status,"missing_COMSOL_mode_shape_count":int(missing)})
    (out/"eigen_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary

