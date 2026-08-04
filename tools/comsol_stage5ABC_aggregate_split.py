#!/usr/bin/env python3
"""Aggregate split Stage 5C calculations and rebuild Stage 5A dashboard.

This script is intentionally lightweight.  The expensive calculations are run as
small Stage3C/Stage3D jobs by scripts/run_stage5ABC_pipeline.sh so sandbox or CI
wall-time limits do not lose all progress.
"""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import comsol_stage5ABC_closure as s5

TARGET={1:1.78,50:1.75,100:1.58,900:1.28,1000:1.24,8000:0.80}


def main():
    split=ROOT/'outputs/stage5C_domain_coil_closure_split'
    out=ROOT/'outputs/stage5ABC_figure_magnetics_coil_closure'
    stage5c=out/'stage5C_domain_coil_closure'; stage5c.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in sorted(split.glob('sigma_*')):
        p=d/'blocked_impedance_exact_voltage.csv'
        if not p.exists():
            continue
        sig=float(d.name.split('_')[1])
        df=pd.read_csv(p)
        for _,r in df.iterrows():
            f=int(round(float(r.f_Hz))); t=TARGET[f]
            err=100*(float(r.Lb_mH)-t)/t
            rows.append({'model':'Stage3C_exact_terminal','sigma_S_m':sig,'f_Hz':float(r.f_Hz),'L_mH':float(r.Lb_mH),'target_L_mH':t,'error_percent':err,'Z_abs_ohm':float(r.Zb_abs_ohm),'Z_real_ohm':float(r.Zb_real_ohm),'Z_imag_ohm':float(r.Zb_imag_ohm)})
    all_df=pd.DataFrame(rows).sort_values(['sigma_S_m','f_Hz'])
    all_df.to_csv(stage5c/'stage5C_sigma_sweep_anchors.csv',index=False)
    summary=[]
    for sig,g in all_df.groupby('sigma_S_m'):
        summary.append({'model':'Stage3C_exact_terminal','sigma_S_m':sig,'rmse_error_percent':float(np.sqrt(np.mean(g.error_percent.values**2))),'max_abs_error_percent':float(np.max(np.abs(g.error_percent.values)))})
    sum_df=pd.DataFrame(summary).sort_values('sigma_S_m')
    sum_df.to_csv(stage5c/'stage5C_sigma_sweep_summary.csv',index=False)
    best=sum_df.iloc[sum_df.rmse_error_percent.argmin()].to_dict()
    selected=all_df[all_df.sigma_S_m==best['sigma_S_m']]
    material=all_df[all_df.sigma_S_m==11200000.0]
    selected.to_csv(stage5c/'stage5C_selected_sigma_anchors.csv',index=False)
    material.to_csv(stage5c/'stage5C_material_sigma_anchors.csv',index=False)
    for name,sig in [('selected',int(best['sigma_S_m'])),('material',11200000)]:
        src=split/f'sigma_{sig}'/'blocked_impedance_exact_voltage.csv'
        if src.exists(): shutil.copy(src, stage5c/f'stage5C_{name}_sigma_blocked_impedance.csv')
    cg_rows=[]
    for d in [split/'stage3D_sigma_1500000', split/'stage3D_sigma_11200000']:
        p=d/'stage3D_figure6_comparison.csv'
        if not p.exists():
            continue
        sig=float(d.name.split('_')[-1])
        df=pd.read_csv(p)
        for _,r in df.iterrows():
            cg_rows.append({'model':'Stage3D_conductor_gauge_fixed_current','sigma_S_m':sig,'f_Hz':float(r.f_Hz),'L_mH':float(r.Lb_mH),'target_L_mH':float(r.target_L_mH_visual),'error_percent':float(r.err_percent_vs_visual),'Z_abs_ohm':float(r.Z_abs_ohm),'Z_real_ohm':float(r.Z_real_ohm),'Z_imag_ohm':float(r.Z_imag_ohm)})
    pd.DataFrame(cg_rows).to_csv(stage5c/'stage5C_conductor_gauge_anchors.csv',index=False)
    # plots
    fig,ax=plt.subplots(figsize=(7,4))
    ax.semilogx(sum_df.sigma_S_m,sum_df.rmse_error_percent,marker='o')
    ax.axvline(11200000,linestyle='--',label='material sigma=1.12e7')
    ax.axvline(1500000,linestyle=':',label='sigma_eff=1.5e6')
    ax.axvline(best['sigma_S_m'],linestyle='-.',label=f'best={best["sigma_S_m"]:.3g}')
    ax.set_xlabel('soft-iron conductivity in scalar Aphi model [S/m]')
    ax.set_ylabel('Figure 6 anchor RMSE [%]')
    ax.set_title('Stage 5C conductivity closure sweep')
    ax.grid(True,which='both',alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(stage5c/'stage5C_sigma_sweep_rmse.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    freqs=selected.f_Hz.values
    ax.semilogx(freqs,[TARGET[int(round(f))] for f in freqs],'o-',label='COMSOL PDF Fig.6 anchors')
    ax.semilogx(freqs,selected.L_mH.values,'s-',label=f'Stage3C split best sigma={best["sigma_S_m"]:.3g}')
    if not material.empty:
        ax.semilogx(material.f_Hz.values,material.L_mH.values,'^-',label='Stage3C material sigma=1.12e7')
    cg=pd.DataFrame(cg_rows)
    if not cg.empty:
        for sig,lab in [(1500000.0,'Stage3D sigma_eff'),(11200000.0,'Stage3D material sigma')]:
            gg=cg[cg.sigma_S_m==sig]
            if not gg.empty: ax.semilogx(gg.f_Hz,gg.L_mH,'x-',label=lab)
    ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('blocked inductance [mH]')
    ax.set_title('Stage 5C Figure 6 closure anchors')
    ax.grid(True,which='both',alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(stage5c/'stage5C_figure6_closure_anchors.png',dpi=180); plt.close(fig)
    # production refined baseline from Stage3C locked output
    comp=pd.read_csv(ROOT/'outputs/stage3C_terminal_baseline/stage3C_figure6_comparison.csv')
    prod=[]
    for _,r in comp.iterrows():
        prod.append({'model':'Stage3C_refined_exact_terminal_sigma_eff_locked','sigma_S_m':1500000.0,'f_Hz':float(r.f_Hz),'L_mH':float(r.exact_voltage_sigma_eff_mH),'target_L_mH':float(r.COMSOL_Fig6_visual_mH),'error_percent':float(r.sigma_eff_error_percent),'material_sigma_L_mH':float(r.exact_voltage_material_sigma_mH),'material_sigma_error_percent':float(r.material_sigma_error_percent)})
    pd.DataFrame(prod).to_csv(stage5c/'stage5C_production_refined_sigma_eff_anchors.csv',index=False)
    rmse=float(np.sqrt(np.mean([p['error_percent']**2 for p in prod])))
    maxe=float(np.max(np.abs([p['error_percent'] for p in prod])))
    mat_rmse=float(np.sqrt(np.mean([p['material_sigma_error_percent']**2 for p in prod])))
    mat_max=float(np.max(np.abs([p['material_sigma_error_percent'] for p in prod])))
    summary_obj={'stage':'Stage 5C Domain Coil / eddy-current closure','mesh':'meshes/comsol_geometry_polyline_coarse_2p5mm.msh for split sweep; refined Stage3C locked output for production baseline','static_source':'B_inverse raw-closure settings','frequency_anchors_Hz':list(TARGET.keys()),'sigma_grid_S_m':sum_df.sigma_S_m.tolist(),'split_best_anchor_sweep':best,'selected_sigma_S_m':float(best['sigma_S_m']),'material_sigma_S_m':11200000.0,'material_to_selected_sigma_factor':float(best['sigma_S_m']/11200000.0),'selected_anchor_rows':selected.to_dict(orient='records'),'material_anchor_rows':material.to_dict(orient='records'),'conductor_gauge_anchor_rows':cg_rows,'production_refined_baseline':{'source':'outputs/stage3C_terminal_baseline/stage3C_figure6_comparison.csv','model':'Stage3C refined exact voltage terminal + sigma_eff=1.5e6 S/m','sigma_eff_S_m':1500000.0,'rmse_error_percent':rmse,'max_abs_error_percent':maxe,'material_sigma_rmse_error_percent_same_refined_run':mat_rmse,'material_sigma_max_abs_error_percent_same_refined_run':mat_max,'rows':prod},'interpretation':['material sigma=1.12e7 S/m over-damps high-frequency blocked inductance in the present scalar Aphi reproduction','sigma_eff=1.5e6 S/m is the COMSOL-like eddy coefficient for this scalar-Aphi baseline, not a material replacement','Stage3C exact global terminal remains COMSOL-like HarmonicLoss=false baseline; Stage3D conductor/gauge remains extended branch'],'closure_status':'PASS_ENGINEERING_REFINED_BASELINE' if rmse<=10 else 'WARN'}
    (stage5c/'stage5C_domain_coil_closure_summary.json').write_text(json.dumps(summary_obj,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Stage 5C Domain Coil / Eddy-Current Closure','','## 计算方式','','- 为避免单次超时，conductivity sweep 按 sigma 分块运行；本脚本只聚合结果。','- split sweep mesh: `meshes/comsol_geometry_polyline_coarse_2p5mm.msh`','- production baseline: `outputs/stage3C_terminal_baseline/stage3C_figure6_comparison.csv`','','## 结论','',f'- split best scalar-Aphi conductivity: {best["sigma_S_m"]:.6g} S/m',f'- production sigma_eff: 1.5e6 S/m',f'- refined production RMSE: {rmse:.3f} %',f'- refined production max abs error: {maxe:.3f} %',f'- material sigma refined RMSE: {mat_rmse:.3f} %',f'- material sigma refined max abs error: {mat_max:.3f} %','','## 解释','','material sigma=1.12e7 S/m 在当前 scalar Aphi 复现中使高频 blocked inductance 过度下降。Stage 5C 的闭合方式是把 `Stage3C exact global terminal + sigma_eff=1.5e6 S/m` 锁为 COMSOL-like baseline，同时保留 material-sigma 和 Stage3D conductor/gauge 为诊断分支。']
    (stage5c/'STAGE5C_DOMAIN_COIL_CLOSURE_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    # Stage5A dashboard
    stage5a=out/'stage5A_reference_dashboard'; stage5a.mkdir(parents=True,exist_ok=True)
    b_path=out/'stage5B_raw_magnetics_closure/refined_B_inverse_iter35/stage5B_raw_magnetics_summary.json'
    if b_path.exists():
        b=json.loads(b_path.read_text(encoding='utf-8'))
    else:
        s2_path=out/'stage5B_raw_magnetics_closure/refined_B_inverse_iter35/stage2_magnetics_summary.json'
        s2=json.loads(s2_path.read_text(encoding='utf-8'))
        raw=s2['summary']['BL_raw_N_per_A']
        b={'stage':'Stage 5B raw magnetics closure','mesh':s2['mesh'],'settings':s2['settings'],'summary':s2['summary'],'target_BL_N_per_A':10.48,'raw_BL_error_percent':100*(raw-10.48)/10.48,'closure_status':'PASS' if abs(100*(raw-10.48)/10.48)<=5 else 'FAIL'}
        b_path.write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
    c_dash={**summary_obj,'selected_anchor_rows':prod,'best_anchor_sweep':{'sigma_S_m':1500000.0,'rmse_error_percent':rmse,'max_abs_error_percent':maxe}}
    dashboard=s5.build_dashboard(stage5a,b,c_dash)
    overall={'stage':'Stage 5A-5C completed','status':'completed','stage5A_dashboard_rows':len(dashboard),'stage5B_raw_BL_N_per_A':b['summary']['BL_raw_N_per_A'],'stage5B_raw_BL_error_percent':b['raw_BL_error_percent'],'stage5C_production_rmse_percent':rmse,'stage5C_material_sigma_rmse_percent':mat_rmse,'stage5C_sigma_eff_S_m':1500000.0}
    (out/'STAGE5ABC_SUMMARY.json').write_text(json.dumps(overall,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# Stage 5A–5C 一次性计算闭合报告','','## 结论','','- Stage 5A：已建立 PDF Figure 3–12 数字化锚点和自动误差 dashboard。','- Stage 5B：已复算 raw magnetics，B_inverse 后 raw BL 不再依赖 calibration。','- Stage 5C：已复算 conductivity sweep，并用 refined Stage3C sigma_eff 锁定 Figure 6 COMSOL-like 生产基线。','', '## Stage 5B 关键结果','',f"- raw BL = {b['summary']['BL_raw_N_per_A']:.9f} N/A",f"- BL target = 10.480000000 N/A",f"- error = {b['raw_BL_error_percent']:.6f} %",'', '## Stage 5C 关键结果','',f'- production baseline: Stage3C refined exact voltage terminal + sigma_eff=1.5e6 S/m',f'- Figure 6 RMSE = {rmse:.3f} %',f'- Figure 6 max abs error = {maxe:.3f} %',f'- material sigma=1.12e7 S/m RMSE = {mat_rmse:.3f} %',f'- material sigma=1.12e7 S/m max abs error = {mat_max:.3f} %','', '## Stage 5A 自动 Dashboard','','- `stage5A_reference_dashboard/figure_error_dashboard.csv`','- `stage5A_reference_dashboard/figure_error_dashboard.md`','- `stage5A_reference_dashboard/figure_error_dashboard.png`']
    (out/'STAGE5ABC_CLOSURE_REPORT_CN.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(overall,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
