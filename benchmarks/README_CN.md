# 频域 benchmark 精选快照

本目录从 `loudspeakerFEM_current_20260717` 的完整分析区中挑选了适合公开发布的、
体积较小且有明确口径的机器可读结果。快照日期为 **2026-07-19**；它们用于审计、
回归对比和后续研究，不是运行时输入。

## 先看什么

| 子集 | 内容 | 适合回答的问题 |
|---|---|---|
| `frequency_response/v3_126pt/` | 1–8 kHz、126 点 native Python 扫频，以及高频三角化 A/B | V3 mapped center-split 结构网格是否改善了高频响应 |
| `frequency_response/stage35_*.json` | 1–15 kHz 的 Stage34/Stage35 汇总 | 8 kHz 以上高精度路由带来的整体收益和限制 |
| `blocked_impedance/` | native MQS 网格收敛、关键频点和分层指标 | blocked 阻抗、raw native 场与内嵌 surrogate 的口径差异 |
| `acoustic_structure/` | ASB 功率积分交叉验证 | 非共形结构—声学边界积分的误差随频率如何变化 |
| `eddy_current/` | COMSOL Req10 的全局阻抗和分域焦耳损耗 | 涡流损耗/blocked 全局量是否有可审计的参考点 |
| `modal/` | 10 个 COMSOL 模态、Python 配对和 mass-MAC | breakup 以前/以后的模态频率与形状一致性 |
| `pml/` | PML Jacobian、深度分箱和典型波长摘要 | PML 变换坐标与轴对称体积因子的独立检查 |

## 关键指标

V3 126 点（1–8 kHz，COMSOL `Study 2` 作为独立参考）的 full coupled 外场结果为：

- 复数 NRMSE：**1.2192%**；幅值 RMSE：**0.1142 dB**；相位 RMSE：**0.7689°**；
- 4–8 kHz 复数 NRMSE：**3.4761%**；6.3–8 kHz：**2.6439%**；
- total impedance 复数 NRMSE：**1.1825%**。

同一官方 mapped 四边形改变三角化方式的 4–8 kHz 汇总为：center split **3.476%**、
主对角线 **4.255%**、反对角线 **11.937%**、交错对角线 **7.597%**。三个高频变体
的逐频 native 输出在 `v3_126pt/` 中；主对角线只在公开快照中保留报告汇总，没有把
内部运行目录整体带入仓库。

Stage35 另有 1–15 kHz、140 点的最终摘要：全角 RMSE 0.1136 dB、主场 RMSE 0.0730 dB、
复形状 NRMSE 0.5348%、原始复数 NRMSE 1.9337%。这组数据目前只有汇总 JSON，没有逐频
COMSOL 原始参考矩阵，因此不能宣称解压后可以独立重算这些历史数字。

## 必须区分的 blocked 口径

`blocked_impedance/native_mesh_convergence_summary.json` 是当前 native-field 网格收敛
快照：最终混合边界层网格为 **0.4224%**，上一版官方 mapped 网格为 **0.6037%**，旧
粗网格/标量 differential 基线为 **65.415%**。这是最适合作为“自包含 native MQS”结论
的口径。

`blocked_impedance/layered_metrics.json` 中的 blocked **0.0272%** 属于离线 COMSOL 识别
的 `embedded_native_surrogate`，运行时虽然不读取 CSV，但它不是独立 raw-field 预测。
因此 README 和 manifest 不把 0.0272% 当作 native MQS 的统一结论。

## 数据边界与复现边界

- COMSOL 在这些记录中只作为独立 benchmark 或探针来源；Python 生产运行不读取这些
  CSV/JSON，也不需要 COMSOL 可执行程序。
- 本目录不包含 solved MPH、COMSOL 原始 `pext` 参考矩阵、Jphi 全点场、模态全点场、
  图片、日志、checkpoint 或内部 `_analysis` 路径。
- CSV/JSON 中的绝对 Windows/WSL 路径已不随快照发布；来源统一写在本说明和
  `manifest.json` 中。
- 快照结果是历史证据，不等于当前工作树重新运行后的验收。重新计算请使用仓库根目录
  的 `cli.py`、`tools/` 和 `configs/`；输出应写入 `runs/` 或项目外目录。

机器可读目录、来源和口径见 [`manifest.json`](manifest.json)。
