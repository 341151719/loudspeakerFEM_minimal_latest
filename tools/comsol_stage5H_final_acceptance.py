#!/usr/bin/env python3
"""Stage 5H final acceptance matrix and package summary generator.

This script does not run new FEM solves. It consolidates computed Stage 5A--5G
CSV/JSON results into a Figure 3--12 acceptance matrix and final technical report.
"""
from __future__ import annotations
import csv, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage5H_final_acceptance"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def load_json(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)

stage5a = read_csv(ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/stage5A_reference_dashboard/figure_error_dashboard.csv')
stage5dg = read_csv(ROOT / 'outputs/stage5D_to_stage5G_closure/stage5D_to_stage5G_dashboard/stage5D_to_5G_error_dashboard.csv')
summary_abc = load_json(ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/STAGE5ABC_SUMMARY.json')
summary_dg = load_json(ROOT / 'outputs/stage5D_to_stage5G_closure/stage5D_to_5G_summary.json')

# Copy detailed metric rows into one dashboard with source stage.
detailed = []
for r in stage5a:
    rr = dict(r); rr['stage_source'] = '5A-5C'; detailed.append(rr)
for r in stage5dg:
    rr = {
        'figure': r.get('figure',''),
        'metric': r.get('metric',''),
        'target': r.get('target',''),
        'python': r.get('python',''),
        'error': r.get('error',''),
        'error_percent': r.get('error_percent',''),
        'status': r.get('status',''),
        'source_python': 'outputs/stage5D_to_stage5G_closure/stage5D_to_stage5G_dashboard/stage5D_to_5G_error_dashboard.csv',
        'stage_source': r.get('stage','5D-5G')
    }
    detailed.append(rr)

write_csv(OUT / 'stage5H_detailed_metric_dashboard.csv', detailed, ['stage_source','figure','metric','target','python','error','error_percent','status','source_python'])

# Per-figure final acceptance matrix: deliberately conservative.
rows = [
    {
        'figure':'Figure 3',
        'pdf_target':'Magnetic field norm H distribution in magnetic motor; BL = 10.48 N/A from coil-domain integral.',
        'python_evidence':'Stage 5B B_inverse raw magnetics: BL = 10.482177800 N/A; error = 0.020781%; H/B/mu field maps exported.',
        'current_method':'Axisymmetric scalar A_phi magnetostatics, nonlinear soft-iron B-H inverse update, ferrite Br=0.4 T, rebuilt COMSOL-like rz geometry.',
        'final_status':'ACCEPTED',
        'remaining_issue':'Distribution is judged mainly by field topology plus BL anchor; no pixel-level comparison against original COMSOL plot image.'
    },
    {
        'figure':'Figure 4',
        'pdf_target':'Effective relative permeability in pole/top plate, saturation near pole center and high-mu regions elsewhere; peak order about 1.2e3.',
        'python_evidence':'Stage 5A dashboard: mu_r_max = 1187.48 versus 1200-order target; error = -1.04%.',
        'current_method':'Effective mu_r derived from biased nonlinear magnetostatic solution.',
        'final_status':'ACCEPTED',
        'remaining_issue':'COMSOL local differential/effective permeability fields are not bitwise reproduced; comparison remains visual/order based.'
    },
    {
        'figure':'Figure 5',
        'pdf_target':'Induced current density J_phi at 50 Hz and 900 Hz, showing increased skin localization at higher frequency.',
        'python_evidence':'Stage 3D exports figure5_stage3D_conductor_gauge_Jphi_real_50Hz.png and 900Hz; 900 Hz field is surface-localized.',
        'current_method':'Frequency-domain eddy-current A_phi perturbation plus conductor/gauge coil branch.',
        'final_status':'QUALITATIVE_ACCEPTED',
        'remaining_issue':'No image-registration metric against COMSOL color plot; current sign/color scale normalization is project-defined.'
    },
    {
        'figure':'Figure 6',
        'pdf_target':'Blocked coil inductance L_b(f) decreases from about 1.78 mH toward about 0.8 mH by 8 kHz.',
        'python_evidence':'Stage 5C production baseline sigma_eff=1.5e6 S/m: RMSE=9.9735%, max abs error=16.129%; 50/900/1000/8000 Hz anchors PASS, 1/100 Hz WARN.',
        'current_method':'Stage3C exact global voltage terminal, scalar A_phi eddy-current operator, COMSOL-like effective conductivity branch; Stage3D conductor/gauge retained as diagnostic.',
        'final_status':'CONDITIONAL_ACCEPTED',
        'remaining_issue':'Using COMSOL material sigma=1.12e7 S/m gives RMSE=42.23% and max=86.71%; sigma_eff is a calibrated effective branch, not strict material-card equivalence.'
    },
    {
        'figure':'Figure 7',
        'pdf_target':'8000 Hz sound pressure level and displacement distribution showing cone breakup with phase-opposed cone regions.',
        'python_evidence':'Stage 4/5 pipeline exports 8 kHz acoustic/structural response and Stage 5D structural modes; no quantitative 8 kHz surface-pattern acceptance metric yet.',
        'current_method':'Axisymmetric solid FEM + ASB acoustic FEM + recovered Boundary-93 HK exterior pressure.',
        'final_status':'PARTIAL',
        'remaining_issue':'Native P2 solid/acoustic and quantitative breakup-pattern comparison are still missing; 8 kHz response remains mesh/HK sensitive.'
    },
    {
        'figure':'Figure 8',
        'pdf_target':'1 m on-axis sensitivity and phase; preferred operating range roughly 100–1500 Hz is comparatively flat; NRA/no-NRA differs near back-cavity modes.',
        'python_evidence':'Stage 5A dashboard still FAILS most SPL anchors: 100 Hz -7.11 dB, 500 Hz -11.50 dB, 1000 Hz +5.44 dB, 1500 Hz +5.28 dB; 5000 Hz passes.',
        'current_method':'Full ASB response with Boundary 93 recovered-HK pext-like postprocessing; with/without NRA branch available.',
        'final_status':'NOT_ACCEPTED',
        'remaining_issue':'Midband flatness is not closed; likely coupled issues in motional impedance, structural damping/calibration, ASB geometry, acoustic radiation load, and low/mid-frequency exterior-field normalization.'
    },
    {
        'figure':'Figure 9',
        'pdf_target':'Lossless back-cavity pressure around 600/630 Hz showing phase/mode switch when NRA is disabled.',
        'python_evidence':'Stage 5F dense sweep: 600 Hz with-minus-without = +3.600 dB, 610 Hz = -1.521 dB, 630 Hz = -1.562 dB; localized mode effect detected.',
        'current_method':'Domain 8/22 strict slit equivalent using parallel-plate thermoviscous complex density/bulk modulus; no-NRA branch disabled.',
        'final_status':'PARTIAL_ACCEPTED',
        'remaining_issue':'Effect is localized but the exact PDF lossless sharp resonance shape is not fully reproduced; moving-wall NRA compatibility and reconstructed back-cavity geometry remain uncertain.'
    },
    {
        'figure':'Figure 10',
        'pdf_target':'Total electric impedance: DC 5.6 Ω, strong mechanical resonance peak near 50 Hz, mostly 6.3–10.4 Ω from 100 Hz–1 kHz, inductive rise above 1 kHz.',
        'python_evidence':'Stage 5A: 1/200/1000/8000 Hz anchors PASS; 50 Hz absZ=9.59 Ω versus ~32 Ω target FAIL; 100 Hz also FAIL.',
        'current_method':'Z_total computed from blocked Z_b plus motional feedback through reduced BL/back-EMF and acoustic-structure response.',
        'final_status':'NOT_ACCEPTED',
        'remaining_issue':'50 Hz motional impedance peak is much too small; full Lorentz/back-EMF/electromagnetic/structural-acoustic global coupling and damping distribution need rework.'
    },
    {
        'figure':'Figure 11',
        'pdf_target':'Main structural modes: first mode around 53.237 Hz; first rotationally symmetric breakup around 2347.4 Hz; higher modes around 2914.9 and 3553.9 Hz.',
        'python_evidence':'Stage 5D suspension_E_scale=0.79156: errors -1.63%, -3.51%, +3.31%, -0.73% for the four anchors.',
        'current_method':'Refined-P1 / P2-surrogate axisymmetric solid FEM, domain 20/25 suspension stiffness calibration, fixed boundaries 81/85.',
        'final_status':'CONDITIONAL_ACCEPTED',
        'remaining_issue':'Frequencies are closed, but method is calibrated surrogate rather than native COMSOL quadratic structural elements and exact material library behavior.'
    },
    {
        'figure':'Figure 12',
        'pdf_target':'Directivity contour from -90° to 90° at 1 m, normalized to 0° direction; high-frequency narrowing and lobes/nulls visible.',
        'python_evidence':'Stage 5G directivity matrix covers 20 Hz–8 kHz and -90°–90° at 1° step; beam widths: 1000 Hz 70°, 5000 Hz 32°, 8000 Hz 64°.',
        'current_method':'Boundary 93 recovered-gradient HK exterior field, relative normalization to 0°.',
        'final_status':'PARTIAL_ACCEPTED',
        'remaining_issue':'Full contour exists, but high-frequency deep nulls/side lobes remain mesh-sensitive; refined/P2 full ASB and COMSOL pext kernel equivalence are not complete.'
    },
]
fields = ['figure','pdf_target','python_evidence','current_method','final_status','remaining_issue']
write_csv(OUT / 'stage5H_final_acceptance_matrix.csv', rows, fields)

status_counts = {}
for r in rows:
    status_counts[r['final_status']] = status_counts.get(r['final_status'], 0) + 1

summary = {
    'stage':'Stage 5H final acceptance matrix and package',
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'source_project': str(ROOT),
    'figure_rows': len(rows),
    'status_counts': status_counts,
    'stage5A_to_5C_summary': summary_abc,
    'stage5D_to_5G_status': {
        '5D': summary_dg['5D']['status'],
        '5E': summary_dg['5E']['status'],
        '5F': summary_dg['5F']['status'],
        '5G': summary_dg['5G']['status'],
    }
}

# Compute selected artifact hashes for reproducibility.
artifacts = [
    ROOT / 'outputs/stage5ABC_figure_magnetics_coil_closure/STAGE5ABC_SUMMARY.json',
    ROOT / 'outputs/stage5D_to_stage5G_closure/stage5D_to_5G_summary.json',
    OUT / 'stage5H_final_acceptance_matrix.csv',
    OUT / 'stage5H_detailed_metric_dashboard.csv',
]
summary['artifact_sha256'] = {}
for p in artifacts:
    if p.exists():
        summary['artifact_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

(OUT / 'stage5H_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

# Markdown reports.
md = []
md.append('# Stage 5H 最终 Figure 3–12 Acceptance Matrix 与版本说明\n')
md.append('## 1. 总结论\n')
md.append('当前项目已经不是早期等效活塞/箱体声学模型，而是一个围绕 COMSOL Loudspeaker Driver — Frequency-Domain Analysis 重建的 Python 轴对称频域 FEM 项目。Stage 5H 的保守结论是：电磁静态场和结构模态已达到图级锚点闭合；blocked inductance、NRA、directivity 达到条件/部分接受；1 m sensitivity 和 50 Hz 总电阻抗峰仍未达到最终 COMSOL 等价。\n')
md.append('状态统计：\n\n')
for k,v in status_counts.items():
    md.append(f'- {k}: {v}\n')
md.append('\n## 2. Figure 3–12 最终接受矩阵\n\n')
md.append('| Figure | PDF 目标 | 当前 Python 证据 | 实现方法 | 最终状态 | 仍有问题 |\n')
md.append('|---|---|---|---|---|---|\n')
for r in rows:
    md.append(f"| {r['figure']} | {r['pdf_target']} | {r['python_evidence']} | {r['current_method']} | {r['final_status']} | {r['remaining_issue']} |\n")
md.append('\n## 3. 当前版本相对最初 COMSOL 版本的实现方法差异\n\n')
md.append('### 3.1 COMSOL 原始模型\n')
md.append('- 几何：COMSOL 使用 `loudspeaker_driver_geom_sequence.mph` 插入完整 2D axisymmetric rz 几何；包含 air、PML、cone、dust cap、surround、spider、coil、former、pole piece、top plate、magnet、baffle 等。\n')
md.append('- 电磁：`Magnetic Fields / Induction Currents`，soft iron 使用 B-H interpolation，ferrite 使用 Br=0.4 T，coil 为 `Homogenized multiturn` Domain Coil，`N0=100`，`VCoil=linper(V0)`，`HarmonicLoss=false`。\n')
md.append('- 结构：COMSOL `Solid Mechanics`，structural domains 包含 composite/cloth/foam/coil/glass fiber；边界 81/85 固定；composite/glass fiber 使用 isotropic loss factor，cloth/foam 使用 `beta_dK` Rayleigh damping。\n')
md.append('- 声学：COMSOL `Pressure Acoustics`，air/PML；Boundary 93 做 `Exterior Field Calculation`，并使用 `z=z0` symmetric/infinite sound hard boundary；`pext(0,1[m])` 用于 1 m 轴上灵敏度。\n')
md.append('- 窄缝损耗：Domain 8 和 22 使用 Narrow Region Acoustics，slit 高度分别 0.4 mm 和 0.2 mm。\n')
md.append('- 多物理场：COMSOL 通过 `AcousticStructureBoundary` 和 `Magnetomechanics/Lorentz Coupling` 自动组装声-固、电磁-结构耦合。研究包括 Magnetic Fields、Complete Model、Without NRA、Eigenfrequency。\n\n')
md.append('### 3.2 当前 Python Stage 5H 版本\n')
md.append('- 几何：从 COMSOL 导出资料、mphtxt/mphbin 和域/边界选择重建 rz 多域模型；域编号和关键边界命名尽量对应 COMSOL，但不是直接执行 COMSOL 的 `.mph` geometry sequence。\n')
md.append('- 电磁静态：自写 axisymmetric scalar `A_phi` FEM，soft iron B-H 用 B_inverse 形式闭合；Stage 5B raw BL 已到 0.0208% 误差。\n')
md.append('- blocked impedance：Stage 3C exact global voltage terminal 作为 COMSOL-like 生产基线；Stage 3D conductor/gauge 作为线圈内部分布式感应电流诊断分支。当前为 `sigma_eff=1.5e6 S/m` 条件闭合，而不是 material sigma 1.12e7 S/m 严格闭合。\n')
md.append('- 结构：自写 axisymmetric solid FEM；Stage 5D 用 refined-P1/P2-surrogate 加 domain 20/25 悬挂刚度尺度修正，闭合 Figure 11 频率锚点。\n')
md.append('- 声-固耦合：自写 ASB block matrix，压力载荷和法向加速度互馈；但最终完整 Study 2 仍采用先电磁参数提取、再声固响应的 reduced EM coupling，不是 COMSOL 单一全局 mf/acpr/solid/mmcpl 同步 Newton/linearized solve。\n')
md.append('- 外场：Boundary 93 HK/recovered-gradient HK 替代 COMSOL `pext()`；已生成 1 m SPL 和 directivity matrix，但 kernel 与 COMSOL Exterior Field Calculation 不是完全同一实现。\n')
md.append('- NRA：用 parallel-plate slit thermoviscous equivalent parameters 近似 COMSOL Narrow Region Acoustics；domain 8/22 和 slit 高度已对应。\n')
md.append('- 求解策略：为沙盒约束，频率分块、checkpoint、coarse/refined 对比；stable 1 mm full ASB 已记录规模但未用 direct LU 全频求解。\n\n')
md.append('## 4. 仍然没有解决的核心问题\n\n')
md.append('1. **Figure 8 未接受**：1 m sensitivity 在 100–1500 Hz 未达到 PDF 的较平坦响应，低/中频锚点多处偏差 5–12 dB；1 kHz/1.3 kHz 强峰仍不合理。\n')
md.append('2. **Figure 10 未接受**：50 Hz 总阻抗机械峰严重不足，约 9.59 Ω 对 PDF 视觉目标约 32 Ω；说明 motional impedance、back EMF、结构阻尼或声负载耦合仍不等价。\n')
md.append('3. **Domain Coil 严格等价未完成**：COMSOL material sigma=1.12e7 S/m 在当前 scalar A_phi 复现中高频电感下降过强；生产基线依赖 `sigma_eff`。\n')
md.append('4. **P2/native elements 未完成**：Stage 5D 使用 refined-P1/P2 surrogate；不是原生二阶 solid/acoustic 形函数。\n')
md.append('5. **1 mm full ASB 求解器未完成**：`comsol_stable_1mm_05gap.msh` 的 pressure free dofs 约 6.16e4，当前没有 block-preconditioned iterative solver；direct sparse 不再是合理路径。\n')
md.append('6. **Figure 7/12 的图像级匹配仍是部分接受**：已有 breakup/directivity 输出，但缺少和 COMSOL 图像/导出场的逐像素或场积分误差。\n')
md.append('7. **COMSOL 原始几何不是 1:1 执行**：由于没有直接在 Python 中运行 `loudspeaker_driver_geom_sequence.mph`，几何仍是重建/反推模型。\n\n')
md.append('## 5. 下一步建议：Stage 6 或 Stage 5I\n\n')
md.append('- 优先级 1：重构 reduced EM coupling，检查 `Fe = BL*V/Zb - BL^2*v/Zb` 的速度定义、peak/RMS、相位符号和施力分布，目标是 Figure 10 的 50 Hz impedance peak。\n')
md.append('- 优先级 2：实现 block-preconditioned GMRES/MINRES，用于 1 mm full ASB，不再用 direct LU。\n')
md.append('- 优先级 3：实现 native P2 acoustic/solid elements，并重新计算 Figure 8/10/12。\n')
md.append('- 优先级 4：对 COMSOL PDF 图或 COMSOL 导出数据进行数字化增强，尤其是 Figure 8/10/12，替换当前视觉锚点。\n')
md.append('- 优先级 5：若能获得原始 `.mph` 或 `loudspeaker_driver_geom_sequence.mph`，重新导入真实几何，消除几何反推误差。\n')
(OUT / 'STAGE5H_FINAL_ACCEPTANCE_REPORT_CN.md').write_text(''.join(md), encoding='utf-8')

brief = f'''# Stage 5 最终包说明\n\n本包包含 Stage 3、Stage 4A–4F、Stage 5A–5H 的全部代码、图、CSV、JSON、报告和复跑脚本。Stage 5H 新增最终 Figure 3–12 acceptance matrix，并明确当前 Python 版本与 COMSOL 原始模型的实现差异和剩余问题。\n\n关键入口：\n\n- `outputs/stage5H_final_acceptance/STAGE5H_FINAL_ACCEPTANCE_REPORT_CN.md`\n- `outputs/stage5H_final_acceptance/stage5H_final_acceptance_matrix.csv`\n- `outputs/stage5H_final_acceptance/stage5H_detailed_metric_dashboard.csv`\n- `outputs/stage5H_final_acceptance/stage5H_summary.json`\n- `scripts/run_stage5H_final_acceptance.sh`\n\n最终保守结论：Figure 3/4 接受；Figure 5 定性接受；Figure 6/11 条件接受；Figure 7/9/12 部分接受；Figure 8/10 未接受。\n'''
(OUT / 'STAGE5_FINAL_PACKAGE_README_CN.md').write_text(brief, encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
