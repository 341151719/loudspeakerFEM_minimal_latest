# Agent 导航图：loudspeakerFEM

本页用于快速选入口和文件，不替代 [`README_CN.md`](../README_CN.md) 中的物理定义、生产状态或验收标准。仓库关系和相似/差异清单见 [`PROJECT_RELATIONSHIP_CN.md`](PROJECT_RELATIONSHIP_CN.md)。

## 项目边界

这是二维轴对称扬声器**频域/谐波** FEM 主线。常见入口为 `cli.py`，生产配置为 `configs/best_model.json`。扫频配置会按频段路由到不同离散方案；不能把时域波形问题塞进这个扫频接口。

## 按任务定位

| 任务 | 首要入口 | 继续阅读 |
|---|---|---|
| 频域单点、扫频、生产路由 | `cli.py`、`best_model/coupled_solver.py` | `configs/best_model.json`、`README_CN.md` 第 1、4 节 |
| 磁场、结构、声学基础装配 | `src/loudspeaker_axisym_fem/` | 对应 `stage4*`、`axisym_*`、`vibroacoustic.py` 模块 |
| P2 结构、混合声学、边界恢复、blocked MQS | `best_model/` | `configs/` 与 `README_CN.md` 第 4–8 节 |
| 密闭/开口/倒相箱、热黏性箱体 | `src/loudspeaker_axisym_fem/enclosure_*.py`、`configs/enclosures/` | `docs/ENCLOSURE_*_CN.md`、`docs/enclosure_phase*_handoff.json` |
| 频域误差或 COMSOL 离线比较 | `tools/`、`comsol_exports/`、`benchmarks/` | `benchmarks/README_CN.md`；参考数据不得进入生产运行时校正 |
| FR10 周期扇区三维分支 | `fr10_full360_cyclic/` | `docs/FR10_FULL360_STATUS_CN.md`；这是独立于二维轴对称主线的 3-D 路线 |
| 回归与接口约束 | `tests/` | `README_CN.md` 第 8、10 节；测试通过本身不等于数值正确 |

## 读代码顺序

1. 先读 `README_CN.md` 第 1、4、7、8、10 节，确认生产路由和不能放宽的模型合同。
2. 从 `cli.py` 的目标子命令定位装配入口；主线频域解从 `best_model/coupled_solver.py` 进入。
3. 沿该入口追到 `src/loudspeaker_axisym_fem/` 中的离散与物理实现，再读对应配置和测试。
4. 改动若涉及瞬态、非线性时域或两个仓库共用代码，先看 [`PROJECT_RELATIONSHIP_CN.md`](PROJECT_RELATIONSHIP_CN.md)；不要用邻接仓库内容静默替换本仓库文件。

## 首要事实

- `inputs/` 里的网格和静磁场文件可能是 Python 生产链必需输入，不是可随意清理的结果文件。
- COMSOL 只作离线独立基准；当前 Python 运行不能依赖 COMSOL 许可证或 benchmark 扫频表。
- 历史误差指标只表示原完整项目中的已记录证据；最小包不包含完整原始对照数据，不能据此宣称本机已复算。
- 数值/物理修改的配置隔离、频段覆盖、网格收敛和完整验收要求，以 `README_CN.md` 为准。
