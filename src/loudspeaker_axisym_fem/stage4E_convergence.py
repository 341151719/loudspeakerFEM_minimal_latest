from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .json_utils import write_json


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as fp:
        return list(csv.DictReader(fp))


def write_csv_rows(path: str | Path, rows: list[dict]) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def keyed_by_frequency(rows: Iterable[dict]) -> dict[float, dict]:
    return {_float(r['f_Hz']): r for r in rows}


def response_convergence(coarse_csv: str | Path, refined_csv: str | Path) -> list[dict]:
    coarse=keyed_by_frequency(read_csv_rows(coarse_csv))
    refined=keyed_by_frequency(read_csv_rows(refined_csv))
    keys=sorted(set(coarse) & set(refined))
    fields=['SPL_1m_hk_dB','SPL_1m_piston_dB','hk_minus_piston_dB','Z_abs_ohm','Z_real_ohm','Z_imag_ohm','Zm_abs_N_s_m','coil_power_W','acoustic_efficiency_percent']
    rows=[]
    for f in keys:
        out={'f_Hz':f}
        for field in fields:
            c=_float(coarse[f].get(field)); r=_float(refined[f].get(field))
            out[f'{field}_coarse']=c
            out[f'{field}_refined']=r
            out[f'{field}_delta_refined_minus_coarse']=r-c
            if field.startswith('Z_') and math.isfinite(c) and abs(c)>1e-12:
                out[f'{field}_relative_delta_percent']=100.0*(r-c)/abs(c)
        rows.append(out)
    return rows


def directivity_convergence(coarse_csv: str | Path, refined_csv: str | Path) -> tuple[list[dict], list[dict]]:
    c=read_csv_rows(coarse_csv); r=read_csv_rows(refined_csv)
    ck={(round(_float(x['f_Hz']), 9), round(_float(x['angle_deg']), 9)): x for x in c}
    rk={(round(_float(x['f_Hz']), 9), round(_float(x['angle_deg']), 9)): x for x in r}
    keys=sorted(set(ck) & set(rk))
    rows=[]
    byf: dict[float, list[dict]]={}
    for k in keys:
        f,ang=k; cr=ck[k]; rr=rk[k]
        row={
            'f_Hz':f, 'angle_deg':ang,
            'SPL_coarse_dB':_float(cr['SPL_dB']), 'SPL_refined_dB':_float(rr['SPL_dB']),
            'relative_coarse_dB':_float(cr['relative_dB']), 'relative_refined_dB':_float(rr['relative_dB']),
        }
        row['SPL_delta_refined_minus_coarse_dB']=row['SPL_refined_dB']-row['SPL_coarse_dB']
        row['relative_delta_refined_minus_coarse_dB']=row['relative_refined_dB']-row['relative_coarse_dB']
        rows.append(row); byf.setdefault(f, []).append(row)
    summary=[]
    for f,rs in sorted(byf.items()):
        spl=np.asarray([x['SPL_delta_refined_minus_coarse_dB'] for x in rs], dtype=float)
        rel=np.asarray([x['relative_delta_refined_minus_coarse_dB'] for x in rs], dtype=float)
        summary.append({
            'f_Hz':f,
            'n_angles':len(rs),
            'SPL_max_abs_delta_dB':float(np.nanmax(np.abs(spl))),
            'SPL_rms_delta_dB':float(np.sqrt(np.nanmean(spl*spl))),
            'relative_max_abs_delta_dB':float(np.nanmax(np.abs(rel))),
            'relative_rms_delta_dB':float(np.sqrt(np.nanmean(rel*rel))),
        })
    return rows, summary


def nra_delta_rows(with_csv: str | Path, without_csv: str | Path, label: str) -> list[dict]:
    a=keyed_by_frequency(read_csv_rows(with_csv)); b=keyed_by_frequency(read_csv_rows(without_csv))
    rows=[]
    for f in sorted(set(a)&set(b)):
        rows.append({
            'mesh_label':label,
            'f_Hz':f,
            'SPL_with_NRA_dB':_float(a[f]['SPL_1m_hk_dB']),
            'SPL_without_NRA_dB':_float(b[f]['SPL_1m_hk_dB']),
            'NRA_delta_with_minus_without_dB':_float(a[f]['SPL_1m_hk_dB'])-_float(b[f]['SPL_1m_hk_dB']),
            'Z_with_NRA_abs_ohm':_float(a[f]['Z_abs_ohm']),
            'Z_without_NRA_abs_ohm':_float(b[f]['Z_abs_ohm']),
        })
    return rows


def summarize_response(rows: list[dict]) -> dict:
    out={}
    if not rows:
        return out
    for field in ['SPL_1m_hk_dB_delta_refined_minus_coarse','SPL_1m_piston_dB_delta_refined_minus_coarse','Z_abs_ohm_delta_refined_minus_coarse','hk_minus_piston_dB_delta_refined_minus_coarse']:
        vals=np.asarray([_float(r[field]) for r in rows if field in r], dtype=float)
        out[field+'_max_abs']=float(np.nanmax(np.abs(vals))) if vals.size else None
        out[field+'_rms']=float(np.sqrt(np.nanmean(vals*vals))) if vals.size else None
    return out


def write_markdown_report(path: str | Path, *, response_rows: list[dict], directivity_summary: list[dict], nra_rows: list[dict], meta: dict) -> None:
    def md_table(rows, cols):
        if not rows:
            return '_无数据_\n'
        s='|'+'|'.join(cols)+'|\n'+'|'+'|'.join(['---']*len(cols))+'|\n'
        for r in rows:
            vals=[]
            for c in cols:
                v=r.get(c,'')
                if isinstance(v,float):
                    vals.append(f'{v:.6g}')
                else:
                    vals.append(str(v))
            s+='|'+'|'.join(vals)+'|\n'
        return s
    summary=summarize_response(response_rows)
    text=[]
    text.append('# Stage 4E P1/mesh/HK 外场收敛审查\n')
    text.append('## 结论\n')
    max_hk=summary.get('SPL_1m_hk_dB_delta_refined_minus_coarse_max_abs')
    max_z=summary.get('Z_abs_ohm_delta_refined_minus_coarse_max_abs')
    text.append(f'- 1000/5000/8000 Hz 的 Boundary-93 HK 轴上 SPL coarse→refined 最大变化 `{max_hk:.3f} dB`。\n' if max_hk is not None else '')
    text.append(f'- 总阻抗 |Z| coarse→refined 最大变化 `{max_z:.4f} Ω`，说明电-机回路对网格不敏感；主要敏感项是 Boundary-93 HK 外场。\n' if max_z is not None else '')
    text.append('- refined directivity 已在 1000 Hz 和 5000 Hz 复跑。1000 Hz 方向性几乎完全收敛；5000 Hz 主瓣/常用角度稳定，但远离主瓣的窄零点/旁瓣仍有局部差异。\n')
    text.append('- 64k nodes 的 `comsol_stable_1mm_05gap.msh` Stage-4D full ASB 在当前沙盒 900 s 内未完成单频点，因此 Stage 4E 保留为“coarse 2.5 mm ↔ refined stage3 seed”的两级可复跑收敛；full 1 mm / P2 需要后续分块矩阵复用或迭代解法。\n')
    text.append('\n## 运行元数据\n')
    for k,v in meta.items(): text.append(f'- {k}: `{v}`\n')
    text.append('\n## 响应收敛\n')
    cols=['f_Hz','SPL_1m_hk_dB_coarse','SPL_1m_hk_dB_refined','SPL_1m_hk_dB_delta_refined_minus_coarse','Z_abs_ohm_coarse','Z_abs_ohm_refined','Z_abs_ohm_delta_refined_minus_coarse','hk_minus_piston_dB_delta_refined_minus_coarse']
    text.append(md_table(response_rows, cols))
    text.append('\n## Directivity 收敛摘要\n')
    cols=['f_Hz','n_angles','SPL_max_abs_delta_dB','SPL_rms_delta_dB','relative_max_abs_delta_dB','relative_rms_delta_dB']
    text.append(md_table(directivity_summary, cols))
    text.append('\n## NRA on/off 差异\n')
    show=[r for r in nra_rows if r.get('f_Hz') in (600.0,630.0,1300.0,1000.0,5000.0,8000.0)]
    cols=['mesh_label','f_Hz','SPL_with_NRA_dB','SPL_without_NRA_dB','NRA_delta_with_minus_without_dB']
    text.append(md_table(show, cols))
    text.append('\n## Stage 4E 状态\n')
    text.append('Stage 4E 完成的是：Boundary 93 HK 外场和 NRA 等效模型的两级网格收敛审查工具、分块复跑结果、方向性收敛摘要、以及失败的 1 mm full ASB 成本记录。它没有宣称 COMSOL Study 2 最终定量闭合；P2/1mm 全频需要 Stage 4F 的矩阵复用/迭代求解。\n')
    Path(path).write_text(''.join(text), encoding='utf-8')


def build_stage4E_outputs(outdir: str | Path, coarse_dir: str | Path, refined_dir: str | Path, refined_dir_1000: str | Path | None = None, refined_dir_5000: str | Path | None = None) -> dict:
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    coarse_dir=Path(coarse_dir); refined_dir=Path(refined_dir)
    response=response_convergence(coarse_dir/'stage4D_complete_response.csv', refined_dir/'stage4D_complete_response.csv')
    write_csv_rows(outdir/'stage4E_response_convergence.csv', response)
    nra=[]
    nra += nra_delta_rows(coarse_dir/'stage4D_complete_response.csv', coarse_dir/'stage4D_without_nra_response.csv', 'coarse_2p5mm')
    nra += nra_delta_rows(refined_dir/'stage4D_complete_response.csv', refined_dir/'stage4D_without_nra_response.csv', 'refined_stage3_seed')
    write_csv_rows(outdir/'stage4E_nra_delta_convergence.csv', nra)
    drows=[]; dsum=[]
    if refined_dir_1000:
        rows,summary=directivity_convergence(coarse_dir/'stage4D_directivity_relative.csv', Path(refined_dir_1000)/'stage4D_directivity_relative.csv')
        drows += rows; dsum += summary
    if refined_dir_5000:
        # compare only f=5000 from coarse multi-frequency directivity file.
        rows,summary=directivity_convergence(coarse_dir/'stage4D_directivity_relative.csv', Path(refined_dir_5000)/'stage4D_directivity_relative.csv')
        drows += rows; dsum += summary
    write_csv_rows(outdir/'stage4E_directivity_angle_convergence.csv', drows)
    write_csv_rows(outdir/'stage4E_directivity_summary.csv', dsum)
    meta={
        'coarse_dir':str(coarse_dir),
        'refined_dir':str(refined_dir),
        'refined_directivity_1000_dir':str(refined_dir_1000) if refined_dir_1000 else '',
        'refined_directivity_5000_dir':str(refined_dir_5000) if refined_dir_5000 else '',
        'stage4D_full_1mm_probe':'attempted /mnt/data/stage4E_probe_stable_1000, timed out at 900 s before output',
    }
    summary={'meta':meta,'response_summary':summarize_response(response),'directivity_summary':dsum}
    write_json(outdir/'stage4E_summary.json', summary, indent=2)
    write_markdown_report(outdir/'STAGE4E_CONVERGENCE_REPORT_CN.md', response_rows=response, directivity_summary=dsum, nra_rows=nra, meta=meta)
    return summary
