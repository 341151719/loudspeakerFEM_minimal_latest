# loudspeakerFEM 项目级改进路线

本页用于选择下一项工作；生产物理合同和验收标准仍以 [`README_CN.md`](../README_CN.md) 为准。不要把这里的候选方案当作已验证结论。

## 已落地的运行基础

- `python cli.py plan --freqs 50,6300,12000` 只读取配置和检查输入，输出每个频率实际使用的配置、离散信息和缺失文件；不装配模型、不启动求解。可用 `--config`、`--single-profile`、`--magnetostatic-vtu` 和 `--blocked-impedance-csv` 对照拟运行的命令。
- `solve` 与 `sweep` 在昂贵装配前检查频率和所需输入。串行扫频按需创建各路由的模型；只扫一个频段时不预先创建其他频段的模型。
- `sweep --jobs 0` 的自动并行数同时受系统可用内存和 WSL/container cgroup 剩余内存限制；显式 `--jobs` 仍由用户控制。
- 成功的 `solve` 与 `sweep` 在输出目录写入 `run_manifest.json`，记录命令、Git 提交与工作树状态、频率到配置的映射、合并后的配置哈希和结束时的输入文件哈希。工作树若有未提交修改，仍需单独保存差异。

## 后续工作优先级

| 优先级 | 工作 | 实施入口 | 接受条件 |
|---|---|---|---|
| 1 | 可重复运行与资源控制 | `cli.py`、`best_model/visualization.py` | 扫频中断后能安全续跑；相同输入/配置不重复求解；结果清楚区分成功、失败和部分完成。 |
| 2 | 频域与时域共享代码的变更审计 | `src/loudspeaker_axisym_fem/`、时域仓库 `inputs/frequency_mainline/` | 生成逐文件依赖和差异清单；只同步经时域调用链核对的文件；两个仓库分别记录基线提交。 |
| 3 | 低频到中频的几何/边界一致性 | `src/loudspeaker_axisym_fem/`、`best_model/`、`tools/production_wet_trace_audit.py` | 声学域、NRA 腔、结构界面和 Boundary93 的几何/法向在同一网格层逐项闭合；保持离线 COMSOL 独立验证。 |
| 4 | 高频网格与误差闭环 | `configs/stage35_high_accuracy*.json`、`tools/stage35_*` | 先对 12 kHz 的全声学域细化候选做 L0/L1/L2，再扩展受影响频段；同时报告自由度、耗时、内存、主场/全角/复数指标；15 kHz 不宣称严格网格无关，直到参考网格也收敛。 |
| 5 | 磁场局部误差与模态 | `best_model/native_blocked_coil.py`、`best_model/eigenmodes.py` | `Jphi` 局部场、复阻抗、损耗和 skin depth 一起收敛；Figure 11 的模态用形状与 MAC 配对，不能只按频率排序。 |
| 6 | 箱体和三维分支的能力边界 | `src/loudspeaker_axisym_fem/enclosure_*.py`、`fr10_full360_cyclic/` | 每条路线单独给出频段、激励、边界和误差证据；二维轴对称与 3-D cyclic/Bloch 的结果不混称。 |

## 实施顺序

1. 先运行 `plan`，确认当前生产路由和输入；在新的 `runs/` 子目录运行基线，保存 manifest 与机器数据。
2. 一次只改变一个物理或离散因素，使用独立诊断配置。对照未修改的生产配置，按 [`README_CN.md`](../README_CN.md) 第 8 节完成频段、网格和物理量矩阵。
3. 只有多指标净改善且回归通过，才考虑修改 `configs/best_model.json`。未完成收敛时保留诊断状态，并明确未通过项。
4. 修改与时域仓库共用的文件前，按 [`PROJECT_RELATIONSHIP_CN.md`](PROJECT_RELATIONSHIP_CN.md) 核对快照依赖；频域与时域分别提交。
