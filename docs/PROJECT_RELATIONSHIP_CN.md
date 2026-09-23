# 两个扬声器 FEM 仓库：相似点、区别和代码边界

本文件与 [`loudspeakerTimeFEM` 的同名说明](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest/blob/main/docs/PROJECT_RELATIONSHIP_CN.md) 互相对应。两个仓库都是独立、可单独安装的 Python FEM 工程；它们研究相同类型的扬声器，但解不同类型的问题，不能把各自的验证结果当作同一模型的等价复现。

## 相似点

- 都针对二维轴对称扬声器多物理 FEM，关注电磁驱动、结构振动和声学响应。
- 都保留 Python 默认链所需的输入文件，提供独立安装/自检/测试入口。
- 两边的 COMSOL 程序和结果用于独立离线验证；Python 生产链不应依赖 COMSOL 运行时或借用 COMSOL 结果作校正。
- 都要求把诊断配置与生产配置隔离，并把历史验证数字视为既有证据，而不是新环境自动重现的结果。

## 区别与任务归属

| 维度 | `loudspeakerFEM` | `loudspeakerTimeFEM` |
|---|---|---|
| 求解类型 | 频域/谐波复数相量；单点与扫频 | 显式时域响应和波形；含非线性瞬态路线 |
| 主关注 | 频率路由、宽频声压/指向性、结构与声学离散、NRA/PML/外场、原生 blocked MQS、模态及箱体 | 电路-结构-声学时域耦合、非线性 BL/磁律、悬挂 Kms ROM、瞬态辐射与时域误差 |
| 常用入口 | `cli.py solve/sweep`；`configs/best_model.json` | `cli.py inspect/run`；README 记录的生产配置为 `configs/transient_70Hz_nonlinear_comsol_physical_abc.json` |
| 主要代码 | `best_model/` 与 `src/loudspeaker_axisym_fem/` | `src/loudspeaker_time_fem/`，加上 `inputs/frequency_mainline/` 的频域依赖快照 |
| 特有分支 | `fr10_full360_cyclic/` 是四周期扇区 3-D 路线；enclosure FEM | tensor co-energy pilot 是未通过固定网格门槛的诊断路线；不能当生产磁律 |

两类结果只有在几何、材料、驱动幅值/波形、边界条件、网格、频率/时间窗和误差定义都匹配时才能比较。频域单点与时域波形不是可互换的验收指标。

## 依赖关系和同步规则

`loudspeakerTimeFEM` 在 `inputs/frequency_mainline/` 中包含一份频域代码及运行输入快照。`src/loudspeaker_time_fem/config.py` 优先选择这份仓库内快照，`src/loudspeaker_time_fem/model.py` 从其 `src/`、`best_model/` 导入结构、声学和网格实现。因此，时域项目默认可独立运行；它不会读取相邻目录中的频域仓库，也不是 Git 子模块。

本次对照的检出基线为：频域仓库 `main@6f18365`，时域仓库 `main@4266d3f`。在这两个检出版本中，快照与旁边频域工作树**不是完全相同的树**：频域仓库多出 enclosure acoustics/geometry/schema/topology/validation 和 `production_wet_trace.py`；`enclosure_models.py` 不同；`best_model/coupled_solver.py`、`p2_axisym_solid.py`、`visualization.py` 不同。未列出的相同文件也不代表以后会持续同步。

更新的逐文件与静态导入闭包审计见 [`TIME_SNAPSHOT_AUDIT_CN.md`](TIME_SNAPSHOT_AUDIT_CN.md)，其基线单独记录为频域 `ebc97e2`、时域 `f1c2f5b`。时域仓库的[函数级核对说明](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest/blob/main/docs/FREQUENCY_SNAPSHOT_AUDIT_CN.md)记录了为什么本次没有同步 `p2_axisym_solid.py`。

因此：

1. 频域功能的源代码改动先落在 `loudspeakerFEM`；时域项目是否需要同步其中某个被导入文件，要按具体依赖逐文件判断。
2. 若修改会影响时域项目，分别更新和提交两个仓库；在时域仓库确认使用的是预期的 `inputs/frequency_mainline/` 副本。
3. 禁止以整目录覆盖方式同步快照。两个副本有不同职责和本地改动，覆盖会丢掉代码或依赖。
4. 只改时域时间积分、非线性磁律或时域边界时，留在 `loudspeakerTimeFEM`，不要反向改写频域生产求解器。

## 互相指向

- 频域仓库：[`341151719/loudspeakerFEM_minimal_latest`](https://github.com/341151719/loudspeakerFEM_minimal_latest)。本地并排克隆名：`../loudspeakerFEM_minimal_latest`（相对于时域仓库）。
- 时域仓库：[`341151719/loudspeakerTimeFEM_minimal_latest`](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest)。本地并排克隆名：`../loudspeakerTimeFEM_minimal_latest`（相对于频域仓库）。

请在目标仓库自己的 Git 根目录执行检查、提交和推送；两边的分支、工作区和提交历史彼此独立。
