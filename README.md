# loudspeakerFEM minimal latest

二维轴对称扬声器频域多物理 FEM 的最小独立源码包，包含磁场、结构、声学、NRA/PML、外场和 COMSOL 离线验证工具，适合 AI 读取、分析和继续开发。

## Start here

完整的中文交接说明在 [`README_CN.md`](README_CN.md)。它记录了生产路由、固定物理合同、Stage29–Stage35 的合并状态、历史验证指标以及尚未闭环的频段和网格问题。版本冻结日期为 **2026-08-01**。

## FR10 full-360 3-D branch

[`fr10_full360_cyclic/`](fr10_full360_cyclic/) on `feature/fr10-full360-cyclic-3d` extends the loudspeaker to a four-sector cyclic/Bloch full-360 3-D P2/local-ASB structural plus front/rear Helmholtz FEM. It exports full-360 VTU fields, 3-D PNG views, and circumferential-order diagnostics. Run `python cli.py fr10-full360 --freq 90 500 1000 2000`; see the [FR10 status note](docs/FR10_FULL360_STATUS_CN.md) for recorded results and limits. The outer boundary is first-order Sommerfeld Robin (not PML), the electrical drive is equivalent `Bl/Rdc/Le` (not full 3-D MQS), and internal dimensions are engineering assumptions rather than manufacturer CAD.

这个仓库不包含完整历史扫频归档、图片、checkpoint、已求解 COMSOL MPH 文件或虚拟环境；`inputs/` 中保留的是 Python 主链必需的网格、几何和静磁偏置场输入，`benchmarks/` 只保留经过筛选的紧凑机器可读快照。

## What this project contains

- `src/loudspeaker_axisym_fem/`：轴对称磁场、结构、声学、NRA、PML、外场和耦合实现；
- `best_model/`：当前生产装配、P2 结构、mapped 声学和 blocked MQS 路线；
- `configs/`：生产路由及 Stage34/35 诊断配置；当前生产入口为 `configs/best_model.json`；
- `inputs/`：开箱运行所需的网格、几何和静磁场输入；
- `comsol_exports/`：有 COMSOL 许可证时使用的离线 Java/Python benchmark 导出；
- `benchmarks/`：精选的 native 扫频、收敛、耦合、损耗、模态和 PML 快照；
- `tests/`、`self_test.py` 和 `tools/`：测试、验收、收敛诊断和报告工具。

Python 主链不在运行时读取 COMSOL 扫频结果；COMSOL 只作为独立 benchmark。COMSOL 许可证和已求解模型不在仓库中。

## Installation and first checks

要求 Python 3.11+。在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python cli.py self-test
python -m pytest -q
```

建议按低、中、高频顺序做真实但独立的冒烟算例，结果写入新建的 `runs/` 或项目外目录，不要写入 `inputs/`：

```bash
python cli.py solve --freq 50 --drive current --current 1 \
  --outdir runs/smoke_50Hz
python cli.py solve --freq 6300 --drive current --current 1 \
  --outdir runs/smoke_6300Hz
python cli.py solve --freq 12000 --drive current --current 1 \
  --outdir runs/smoke_12000Hz
```

完整扫频和高频细化计算可能较重；先完成测试和 50 Hz，再根据任务需要运行 6300/12000 Hz。生产频率会自动路由到不同离散配置，不能为了省时把所有频率强行使用同一配置。

## Current status and limitations

- Stage35 已将 8 kHz 以上的局部结构/声学细化设为默认高频路线；
- 1–15 kHz 的历史指标是已有证据摘要，解压后不会自动重新生成这些数据；
- 15 kHz 网格无关性尚未完全闭环，12 kHz 和局部 `Jphi` 仍有已知误差；
- 少量历史诊断脚本保留了旧 `/mnt/...` 默认路径，运行时必须显式覆盖本机路径；生产 `cli.py` 主链使用仓库内输入。

任何物理或离散修改都必须同时检查低/中/高频、受影响网格层、声学边界条件、blocked 阻抗和与未修改生产配置的回归。详细完成定义和禁止事项见 [`README_CN.md`](README_CN.md)。
