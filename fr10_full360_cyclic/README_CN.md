# FR10 full-360 FEM 交接包（2026-09-02）

## 目的

这是从 `loudspeakerFEM_minimal_latest` 逐步升级得到的 FR10 三维 FEM 工作包。当前已经完成：

- FR10 P2 三维运动系统；
- cone / dustcap / surround 局部曲面 ASB，已淘汰 rank-1 `Sd` 耦合；
- 前、后两个三维 Helmholtz 声场；
- 90 Hz 自由空气共振的 P2/local-ASB 标定；
- quarter P2/local-ASB 的 500 / 1000 / 2000 Hz 基准结果；
- full-360 解空间的 cyclic/Bloch formulation；
- 90° 结构切面矢量旋转周期条件；
- 声学周期切面的复 Bloch 条件；
- k=0,1,2,3 四个 phase class 的代码框架；
- full-360 求解已改为 exact periodic local-ASB trace condensation，不再依赖 nested Krylov。

**注意：旧交接记录中的“500/1000/2000 Hz 尚未完成”和“nested Krylov blocker”是 exact trace condensation 实施前的历史状态；当前基线结果已闭合。**

## 目录

`fr10_full360_cyclic/`
: 当前应继续处理的版本。

- `cyclic_full360_solver.py`：full-360 cyclic/Bloch 主求解器。
- `base_p2_local_solver.py`：P2 + local ASB 基础装配代码。原工作目录缺少这个别名，本交接包已补齐。
- `generate_structural_meshes.py`：按 full360 配置生成结构 sector 网格。
- `gen_periodic_quarter_ac.py`：生成严格周期声学 sector 网格，输出为求解器实际读取的 `meshes/acoustic_base_quarter.msh`。
- `prepare_meshes.sh`：依次生成结构和周期声学网格。
- `run_full360.sh`：500/1000/2000 Hz 基线运行入口。
- `configs/fr10_full360_cyclic.json`：当前参数。

当前提交只筛选导入本分支所需的源码、sector/周期声学网格和 benchmark 快照；未导入重复的基线 ZIP 或 runtime libs。历史基线结果仅作为 benchmark 参考，不随本目录重复打包。

## 运行命令

在项目根目录执行：

```bash
python cli.py fr10-full360 --freq 90 500 1000 2000
python cli.py fr10-full360 --freq 2000 --diagnostic-phases 1 2 3
```

第一条执行 k=0 full-360 基线，第二条执行 k=1/2/3 非镜面对称 phase 诊断。结果不写入 Git；默认输出根目录为仓库同级 `runs/<checkout-name>_fr10_full360`，可用环境变量 `FR10_FULL360_OUTPUT_ROOT` 或 CLI 的 `--outdir` 覆盖。本次交接结果的 `final_baseline/` 与 `phase_diagnostics/` 位于项目外的 `runs/fr10_full360_feature_20260902/`。如需重建网格，可先运行 `fr10_full360_cyclic/prepare_meshes.sh`。

已有结果还可生成 2000 Hz 的复场动画和 1 m 频响：

```bash
python cli.py fr10-animate --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 --frequency 2000 --frames 24 --fps 12
python cli.py fr10-animate --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 --surface-suite --frames 24 --fps 12
python cli.py fr10-animate --results-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902 --complete-assembly-suite --assembly-cad-root /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/reference_cad_20260902/FR10_COMSOL_3D_model --frames 24 --fps 12
python cli.py fr10-response --output /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/frequency_response_1m --reuse-summary /mnt/c/Users/Administrator/Documents/PYTHON2COMSOL/runs/fr10_full360_feature_20260902/final_baseline/run_summary.json
```

首条动画命令输出原有五类声场/点位诊断。`--surface-suite` 在 `animations/membrane_surface_3d/` 输出三种连续膜面 GIF。`--complete-assembly-suite` 在 `animations/complete_loudspeaker_3d/` 使用交接包内层 `VISATON_FR10_COMSOL_3D_baseline.zip` 的原始 STL，显示盆架、磁体、导磁板、极芯、端子及运动组件；六个可见运动 CAD 表面按部件映射 full-360 FEM 位移，颈部胶参与求解但位于内部。固定件当前仅作 CAD 装配显示，不参与结构位移求解。频响输出为 `frequency_response_1m/frequency_response_1m.{png,csv,json}` 和 `frequency_response_1m_2p83Vrms.png`，覆盖 50--2000 Hz 的 18 个频点；1 m 数值由 0.3 m Sommerfeld 球面出射波外推，不是 PML。

## 当前数学模型

结构 cyclic condition：

`u(theta+pi/2) = exp(i*k*pi/2) R90 u(theta)`

声学 cyclic condition：

`p(theta+pi/2) = exp(i*k*pi/2) p(theta)`

其中 `k=0,1,2,3`。四个 phase class 联合张成四 sector 离散的 full-360 解空间，不再使用 quarter mirror BC。

局部 ASB：

`G_ij = integral_Gamma N_u,i . n N_p,j dS`

因此 cone、dustcap、surround 的局部自由度独立耦合到声场，`Sd` 只作为参考值，不控制耦合形状。

## 当前求解方法

旧交接包中的 nested Krylov blocker 已移除。当前 `solve_phase()` 对结构、前声学、后声学体块分别执行 sparse LU，在周期 ASB trace 上形成精确 condensed operator，再以 dense trace LU 求解并回代；所有 local-ASB trace 自由度均保留，没有退回 rank-1 `Sd` 或镜面对称边界。

full-360 输出同时包含四扇区显式重建的结构/声学 VTU、结构位移和外声场 SPL 的 3D PNG，以及周向位移 `m`-energy PNG/CSV。k=1/2/3 的 phase 诊断用于证明非镜面对称三维解空间处于激活状态。

不要为了跑通而退回 rank-1 ASB、P1 薄实体或重新施加镜面对称。

## 物理边界与解释边界

- 声学外边界仍是一阶 spherical Sommerfeld Robin，而不是 PML。
- 电磁部分仍为 `Bl/Rdc/Le` 等效驱动，并非完整 3D MQS 电磁场仿真。
- 内部尺寸含工程假设，不能称为厂家 CAD；相关参数应保持可调。
- 90 Hz `Zmot`、500/1000/2000 Hz SPL 和 phase 诊断的记录值见 [`docs/FR10_FULL360_STATUS_CN.md`](../docs/FR10_FULL360_STATUS_CN.md)。

本目录内的 `meshes/` 保存 sector 结构网格和严格周期声学网格；完整结果在运行时生成到项目外输出根目录，不提交结果文件。历史 `HANDOFF_BLOCKER_AND_NEXT_STEP.txt` 保留改造前记录，当前状态以 [`docs/FR10_FULL360_STATUS_CN.md`](../docs/FR10_FULL360_STATUS_CN.md) 为准。

## 频域摇摆模态与根因分离

`rocking_frequency.py` 在现有三维 cyclic FEM 上增加论文式弱不对称前馈分析：

- `k=0` 电驱动解给出音圈位移和电流；
- `k=1` 解施加归一化的音圈分布单位力矩，得到包含前后声学负载的摇摆柔度；
- 质量、刚度和 Bl 不对称分别生成 `omega^2*Delta_m*X`、`-Delta_k*X` 和 `Delta_Bl_moment*I`；
- 输出各根因力矩、倾角及总倾角的 JSON、CSV 和 PNG。

运行示例：

```bash
python cli.py fr10-rocking --freq 80 90 120 200 300 350 500 \
  --outdir runs/fr10_rocking_frequency
```

默认参数位于 `fr10_full360_cyclic/configs/rocking_modes_frequency.json`，只是量级合理的诊断扰动，并非 FR10 实测缺陷。当前实现采用论文的弱耦合前馈近似：不对称力矩不会反向改变 `k=0` 活塞/电路解，也没有把四个 Bloch phase class 装配为一个含缺陷的全耦合矩阵；电磁驱动仍为等效 `Bl/Rdc/Le`，不是三维 MQS。

单个复数 `k=1` 解表示 `m=1` 摇摆双重态的一支圆极化基；固定空间方向的实摇摆由共轭的 `k=1/k=3` 组合恢复。对当前旋转对称线性基线，两支具有相同的标量摇摆柔度，因此根因幅频分离使用 `k=1` 即可。

实际验证和结果解释见 [`docs/ROCKING_MODES_FREQUENCY_CN.md`](../docs/ROCKING_MODES_FREQUENCY_CN.md)。

该文档还提供已生成的连续振膜和全结构节点 GIF，可直接在 GitHub 中查看不同根因和摇摆形式的可视化结果。
