# FR10 full-360 cyclic 3D 状态说明

日期：2026-09-02  
功能分支：`feature/fr10-full360-cyclic-3d`  
输入交接包：`C:\Users\Administrator\Documents\FR10_full360_cyclic_handoff_20260902.zip`  
交接包 SHA-256：`8057390548f839b67f1b62604a0bac2cea9d8ddc606163694d8025ded9aaba2b`

## 交付范围

本分支把 FR10 扬声器从 quarter/轴对称参考路线延伸为四扇区 cyclic/Bloch 的 full-360 三维有限元模型，包含：

- P2 tetra10 结构运动系统，以及 cone、dustcap、surround 的局部曲面 ASB；
- 前、后两个三维 Helmholtz 声场；local ASB 保留各曲面自由度，不使用 rank-1 `Sd` 耦合；
- 90° 结构切面的矢量旋转周期条件和声学复 Bloch 条件；
- `k=0,1,2,3` 四个 phase class。k=0 是对称电驱动基线，k=1/2/3 是非镜面对称解空间诊断；
- exact periodic local-ASB trace condensation：结构、前声学、后声学体块分别 sparse LU，在 ASB trace 上形成精确 condensed operator，再用 dense trace LU 求解并回代；
- full-360 四扇区显式重建、VTU 场导出、3D PNG 可视化，以及位移周向 Fourier `m`-energy 诊断。

因此，旧交接记录中“未完成的 full-360 数值结果”和“nested Krylov blocker”属于 exact trace condensation 实施之前的历史状态；当前基线已闭合。

## 可复现命令

在项目根目录执行：

```bash
python cli.py fr10-full360 --freq 90 500 1000 2000
python cli.py fr10-full360 --freq 2000 --diagnostic-phases 1 2 3
```

第一条执行 k=0 full-360 基线；第二条在 2000 Hz 执行 k=1、k=2、k=3 诊断。结果不提交到 Git。默认输出根目录为仓库同级 `runs/<checkout-name>_fr10_full360`；可用环境变量 `FR10_FULL360_OUTPUT_ROOT` 或 CLI 的 `--outdir` 覆盖。本次实际最终结果位于项目外：

```text
/mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/final_baseline/
/mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/phase_diagnostics/
```

若要复现到同一输出根目录，可在运行前设置 `FR10_FULL360_OUTPUT_ROOT`；也可将 `--outdir` 指向上述两个结果子目录。不要把运行结果复制回仓库提交。

## 已记录验收数字

90 Hz full-360 k=0 的运动阻抗为：

```text
Zmot = 15.942170167682338+0.00014409395472950233i ohm
```

500/1000/2000 Hz 前轴 1 m SPL 与 quarter P2/local-ASB 参考的差值为：

| 频率 | 前轴 SPL | 相对 quarter SPL 差 |
|---:|---:|---:|
| 500 Hz | 76.063558 dB | -0.001597 dB |
| 1000 Hz | 81.554463 dB | -0.002139 dB |
| 2000 Hz | 82.298474 dB | +0.040290 dB |

这三点的 `Zmot` 最大复数相对误差为 `0.337416%`。

周向 phase 诊断结果：

| phase class | dominant `m` | class fraction | status |
|---:|---:|---:|---|
| k=1 | m=1 | 1 | pass |
| k=2 | m=2 | 1 | pass |
| k=3 | m=3 | 1 | pass |

`class fraction=1` 表示重建场的周向能量完全落在该 phase class 对应的 mod-4 类中；这证明 k=1/2/3 的非镜面对称三维解空间确实被激活，而不是把 quarter 镜面对称结果复制四次。

## 可视化与机器输出

`final_baseline/` 下每个频点包含 JSON summary、压缩的 sector solution，以及显式四扇区重建的：

- 结构 `structure_full360_*.vtu`：`u_real_m`、`u_imag_m`、`u_abs_m`、`u_z_phase_deg`；
- 前/后声场 `acoustic_*_full360_*.vtu`：复压力、幅值、相位和 RMS SPL；
- 结构位移幅值、`Re(u_z)` 和外声场 SPL 的 3D PNG；
- circumferential `m` 能量的 PNG 与 CSV。

`phase_diagnostics/` 下按 `2000Hz_k1`、`2000Hz_k2`、`2000Hz_k3` 保存同类 VTU/PNG/CSV，并在 `diagnostic_summary.json` 汇总 phase class、非镜面对称状态和残差。VTU 可用 ParaView 等工具旋转查看；PNG 适合快速检查三维几何、位移和外声场分布。

## 动画与 1 m 频响

动画和 1 m 频响均从项目根目录运行。对已有的 FR10 full-360 结果，可执行：

```bash
python cli.py fr10-animate \
  --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 \
  --frequency 2000 --frames 24 --fps 12
python cli.py fr10-animate \
  --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 \
  --surface-suite --frames 24 --fps 12
python cli.py fr10-animate \
  --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 \
  --complete-assembly-suite \
  --assembly-cad-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/reference_cad_20260902/FR10_COMSOL_3D_model \
  --frames 24 --fps 12
python cli.py fr10-response \
  --output /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/frequency_response_1m \
  --reuse-summary /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/final_baseline/run_summary.json
```

第一条命令在 `runs/fr10_full360_feature_20260902/animations/2000Hz/` 写出 24 帧、12 fps 的五个 GIF：

- `rocking_vibration_2000Hz_k1.gif`：k=1/m=1 归一化 Bloch 诊断的摇摆振动（形变为可视化放大）；
- `source_propagation_meridional_2000Hz_k0.gif`：k=0 前/后外场 y=0 经向剖面的复声压传播；
- `outer_field_3d_propagation_2000Hz_k0.gif`：k=0 前/后外场的 3D cutaway 瞬时声压；
- `outer_boundary_pressure_2000Hz_k0.gif`：k=0、R=0.3 m 外边界瞬时声压；
- `outer_boundary_pressure_2000Hz_k1_diagnostic.gif`：k=1/m=1 诊断、R=0.3 m 外边界瞬时声压。

动画使用 `exp(+i omega t)` 约定从复数场重建瞬时压力/位移；k=1 项是单位广义力的非镜面对称解空间诊断，不应解释为对称电驱动的绝对幅值。机器可读的动画路径、帧数和色标记录在同目录的 `animation_summary.json`。

第二条命令从 tetra10 外边界提取 surround、cone、dustcap 的连续二次三角面，在 `animations/membrane_surface_3d/` 写出三个 24 帧 GIF：90 Hz k=0 物理电驱动活塞振动、2000 Hz k=0 物理电驱动分割振动，以及 2000 Hz k=1/m=1 单位力摇摆诊断。顶点逐帧按 `x(t)=x0+scale*Re(U exp(i omega t))` 变形；标题给出几何放大倍率，色标始终显示未放大的真实瞬时 `u_z`（micrometre peak）。元数据见 `surface_animation_summary.json`。

第三条动画命令使用交接包内层 `VISATON_FR10_COMSOL_3D_baseline.zip` 的原始部件 STL，在 `animations/complete_loudspeaker_3d/` 生成相同三个工况的完整扬声器装配动画。固定 CAD 包括盆架、上导磁板、铁氧体磁体、后导磁板、极芯、端子板和正负端子；surround、cone、dustcap、spider、former、coil 的 CAD 表面按部件最近节点映射现有 full-360 FEM 复位移。映射距离、CAD 来源、放大倍率与“固定件未参与当前结构求解”的边界均记录在 `complete_assembly_summary.json`。

第四条命令写出 `frequency_response_1m/frequency_response_1m.{png,csv,json}`，并另存标准化曲线 `frequency_response_1m_2p83Vrms.png`。频响覆盖 50--2000 Hz 的 18 个 1/3 倍频程附近频点（50、63、80、90、100、125、160、200、250、315、400、500、630、800、1000、1250、1600、2000 Hz）。主列为 1 V peak（0.70710678 V RMS）；2.83 V RMS 列仅由线性比例换算，不是重新求解。1 m 复声压由 0.3 m 一阶 spherical Sommerfeld 边界按球面出射波外推得到；该边界不是 PML。

本次频响中新增求解的最大块相对残差为 `2.994e-7`，最大后向误差为 `2.632e-20`；复用的既有 `final_baseline` 行仍按原始 summary 保留。

## 物理边界与解释边界

- 声学外边界仍是一阶 Sommerfeld Robin，而非 PML；高频外场精度不能按 PML 结果解释。
- 电磁部分仍采用 `Bl/Rdc/Le` 等效模型；这不是把线圈、磁隙、磁体和铁磁件都离散进来的完整 3D MQS 仿真。
- 模型内部尺寸（例如 surround roll、spider 等产品图未给出的截面）含工程假设，参数可调；不能称为厂家 CAD，也不能据此声称复原了厂家内部几何。
- 本说明确认的是 full-360 结构/声学 FEM、周期解空间、数值凝聚和可视化链路；“完整 3D”不等于等效电磁部分已经升级为 3D MQS。
