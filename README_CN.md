# loudspeakerFEM 最小独立交接包

版本冻结日期：2026-08-01  
主线来源：`00_MAINLINE/loudspeakerFEM_current_20260717` 的 Stage29–Stage35 合并状态。  
交付性质：解压后可独立安装和运行 Python FEM；不包含完整历史结果、图片或 COMSOL 已解模型，精选 benchmark 快照见 [`benchmarks/`](benchmarks/)。

## 1. 新 AI 的执行基准

本项目是二维轴对称扬声器频域多物理 FEM 的当前生产主线。默认入口是：

```text
configs/best_model.json
```

扫频按频率自动路由：低频使用 `configs/fast_p1.json`；3–8 kHz 使用主配置的 mapped 结构/P2 声学路线；8 kHz 以上使用 `configs/stage35_high_accuracy.json`。不得为了省时把所有频率强行统一到一个离散配置后仍称为生产结果。

本包不需要 COMSOL 安装即可运行 Python 主链。COMSOL Java 文件仅用于未来有许可证时重建独立 benchmark，不能在 Python 生产运行时提供校正值。

## 2. 包内容与数据边界

保留：

- `src/loudspeaker_axisym_fem/`：磁场、结构、声学、NRA、PML、外场和耦合基础实现；
- `best_model/`：当前生产装配、P2 结构、混合声学、blocked MQS、Boundary93 恢复和输出；
- `configs/`：生产路由及 Stage34/35 收敛诊断配置；
- `inputs/`：开箱运行不可缺少的几何、网格、磁静态场和少量配置输入；
- `tests/`、`self_test.py`；
- `tools/` 和 `comsol_exports/` 中的源码；
- 本文件，且仅保留这一份项目说明文档。

排除：

- 既有 `runs/`、完整历史扫频 CSV/NPZ、checkpoint 和大体积 benchmark 原始归档；
- PNG/JPG/GIF、普通 VTK/VTU 结果；
- solved MPH、COMSOL 导出结果目录和日志；
- 虚拟环境、pip 包、缓存、编译产物、历史补丁、备份和旧报告。

三个保留的 VTU 是默认 Python 链所需的静磁偏置场输入，并被源码包清洁测试明确列入白名单。它们不是本次附带的扫频/声学完成结果。`Untitled.mphtxt`、`loudspeaker_driver_exported.m` 和 mapped `.msh` 是离线几何/网格输入；运行时不调用 COMSOL。

## 3. 安装与第一次验收

要求 Python 3.11+。在解压目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python cli.py self-test
python -m pytest -q
```

随后运行不绘图的真实 FEM 冒烟算例：

```bash
python cli.py solve --freq 50 --drive current --current 1 \
  --outdir runs/smoke_50Hz
```

再检查生产自动路由的代表频率：

```bash
python cli.py solve --freq 6300 --drive current --current 1 \
  --outdir runs/smoke_6300Hz
python cli.py solve --freq 12000 --drive current --current 1 \
  --outdir runs/smoke_12000Hz
```

12 kHz 计算明显更重。新 AI 第一次接手至少必须完成 50 Hz；涉及生产改动时必须完成低/中/高频三点和完整相关测试。输出写入 `runs/` 或项目外目录，不得写入 `inputs/`。

其他入口：

```bash
python cli.py magnetics --outdir runs/magnetics
python cli.py blocked --freqs 50,1000,8000 --outdir runs/blocked
python cli.py blocked --freqs 50 --raw-field --outdir runs/blocked_raw
python cli.py eigen --n-modes 40 --outdir runs/eigen
python cli.py sweep --freqs comsol_126 --drive voltage --voltage 3.55 \
  --jobs 8 --blas-threads 1 --outdir runs/final126
```

默认 `solve` 需要磁静态 VTU；包内已提供。`magnetics` 可从头重算原生静磁场，不需要 COMSOL。

## 4. 当前生产模型合同

不得无意改变以下合同：

- 结构使用轴对称 P2 三角形；Lorentz 力与 back-EMF 使用共轭耦合；
- 低频 P1，高频 mapped 结构网格；结构/声学非共形时使用 P2 ASB 边积分；
- 8 kHz 以上对结构域 21/25 局部协调细化，并对声学域 2/4/7 使用 P2；
- PML 与物理域界面保持 Boundary93 P1 trace；
- NRA 是原生平行板热黏性模型，禁止 local-transfer 和 log-frequency 校正；
- Boundary93 使用物理侧二次 PPR、准确径向法向和统一的 `exp(+iwt)` / 出射 Green 函数约定；
- blocked impedance 默认是原生同网格 MQS、非线性静磁偏置和完整切线磁阻张量；
- COMSOL 只作离线 benchmark，不得在运行时读取其 CSV/VTU 扫频结果；
- 配置中的历史 surrogate 只作兼容/诊断，生产 `runtime_mode` 必须保持原生场路线。

## 5. 已合并成果

- Stage29：原生平行板热黏性 NRA，移除 local-transfer、对数频率插值和降阶修正；
- Stage30：Boundary93 物理侧二次 PPR、径向法向及一致的 HK 相量/Green 约定；
- Stage31：global-P2/skin-P2 涡流场、分域损耗、skin depth 和 Figure 5 导出链；
- Stage32：Figure 8 全 126 点原生链，禁用 transfer correction，并修复结构高频倍率未传入生产装配；
- Stage33：blocked/eigen CLI、ASB P2 边积分、十模态 MAC、REQ10/11 探针、软铁电导率修复和原生 blocked MQS；
- Stage34：8 kHz 以上二级结构细化，将剩余高频误差定位到纸盆边界；
- Stage35：用全层复数场和因果替换定位结构域 21/25 与声学域 2/4/7 的欠分辨，并将对应局部加密/P2 路线设为 8 kHz 以上默认。

当前生产不是把 COMSOL 场代入 Python 得到的。COMSOL 全场代入只用于定位误差因果层，不允许进入最终产品。

## 6. 已验证指标与正确解释

历史完整证据给出的 1–15 kHz 合并 140 点指标为：

- 全角相对 RMSE 均值 0.1136 dB；
- 主场 RMSE 均值 0.0730 dB；
- 复形状 NRMSE 均值 0.5348%；
- 原始复数 NRMSE 均值 1.9337%；
- 轴上幅值 RMSE 0.1935 dB；
- 轴上相位 RMSE 约 1.033°。

Stage35 高频 15 点相对 Stage34：全角 RMSE 1.0061→0.6754 dB，主场 0.5831→0.4475 dB，复形状 4.2910%→3.2045%，原始复数 5.7741%→4.2553%。

原生 blocked MQS 的 126 点复阻抗 NRMSE 在当前报告口径为约 0.604%。配置内还记录了不同阶段/字段口径的 0.4224% 或约 1.096% 等数值；引用时必须同时说明所用网格、是否 raw field、是否残差闭合及比较定义，禁止挑选最小数字当作统一结论。

本最小包没有携带原始 COMSOL 对照数据，因此上述是历史结论，不是解压后重新复算得到的证据。新结果必须输出自己的机器可读数据，不能引用本段替代验收。

`benchmarks/` 补充了经过筛选的紧凑 CSV/JSON 快照，并单独记录来源、用途和比较口径；
它不是运行时依赖，也不替代完整 COMSOL 参考矩阵。详见 [`benchmarks/README_CN.md`](benchmarks/README_CN.md)。

## 7. 已知未闭环项

- COMSOL 结构网格从 1 mm 到 0.5 mm 在 15 kHz 仍有最多约 1.1583 dB 变化；最上端只能称趋势收敛，不能称严格网格无关；
- 12 kHz Stage35 默认相对 benchmark 仍有约 1.54–1.58 dB 主场误差和约 8.35–8.70% 复形状误差；全声学域再细化可改善，但仅验证单频且成本高，未设为默认；
- 局部 P1 `Jphi` 峰值与 COMSOL 高阶单元仍有约 4.9–9.1% 绝对复 NRMSE；
- 无 NRA 腔模的剩余偏移与声学离散/曲线几何有关；
- Figure 11 breakup 模态的完整 MAC 闭环仍不足；
- 2.12 kHz、13.5 kHz 的深零点对角度偏移敏感，全角最大 dB 不能单独代表主瓣质量；
- 当前 Stage35 高精度路线换取的是自包含与一致性，不应宣称普遍比 COMSOL 更快。

## 8. 修改时固定的验证矩阵

对任何物理或离散修改，至少执行：

1. `python cli.py self-test` 与完整 pytest；
2. 50 Hz、6300 Hz、12000 Hz 三个当前路由的独立算例；
3. 对受影响频段做网格 L0/L1/L2，而不是只比较 Python 与一个 COMSOL 网格；
4. 同时报轴上、全角、主场、能量加权、复形状、原始复数、波束宽度；
5. 结构修改检查位移场、边界运动、Lorentz/back-EMF 共轭和模态；
6. 声学修改检查各声学域、Boundary93 的 `p` 与 `dp/dn`、法向和 HK 符号；
7. blocked 修改检查复阻抗、分域损耗、skin depth、局部 `Jphi` 和静磁偏置场收敛；
8. 与未修改生产配置做完整回归；没有多指标净改善就不升级生产。

## 9. 新研究的图和数据要求

虽然交付包不带历史图，新 AI 在执行研究时必须生成并检查：网格/域/边界编号图；结构位移幅相场；声压幅相场；Boundary93 压力与法向梯度；极坐标指向性；误差随频率曲线；blocked 的阻抗、损耗、skin depth、`Jphi`；模态形状和 MAC；每层网格的自由度、耗时、内存与指标变化。

每张图必须标注频率、配置、网格层、单位和相量约定。每个图必须能追溯到 CSV/JSON/NPZ；机器数据必须包含配置路径、输入哈希、版本日期和运行命令。只给图片或只给最终均值都不算完成。

## 10. 禁止事项与最终完成标准

禁止读取 COMSOL benchmark 表作为生产校正；禁止逐频调材料参数；禁止覆盖 `configs/best_model.json` 做探索；禁止用单点改善替代全频回归；禁止忽略深零点指标定义；禁止把离线 COMSOL 派生几何说成完全从零建模；禁止把测试通过等同于数值正确。

交付一项修改前，必须确保：源码、配置和输入自包含；安装/自检/测试通过；至少一个真实 FEM 求解成功；受影响频段完成收敛和回归；图与机器数据齐全；没有隐藏绝对路径；COMSOL 若参与，仅作为独立 benchmark；所有未通过项在结论中显式保留。
