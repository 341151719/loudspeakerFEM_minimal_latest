# 外部 COMSOL 二维模型基准审计

2026-09-23 检查了并排工作区中的 `../扬声器摇摆模态/COMSOL 3D CASE/2D基准部分/loudspeaker_driver.mph` 和同目录 `loudspeaker_driver.m`。这两个文件**不在本 Git 仓库内**。

| 项目 | 检查结果 |
|---|---|
| MPH SHA-256 | `7b106cb3feb4a1617ad57821048a7a2de3ba4c1c94b175054a544d2e299bbd82` |
| 文件格式和版本 | COMSOL 6.3.0.290 MPH；ZIP 内容完整，压缩包校验无误 |
| COMSOL 实际加载 | 在已安装的 COMSOL 6.3 批处理环境中，从 MPH 副本只读加载成功 |
| 模型结构 | 二维轴对称；`mf` 磁场、`acpr` 压力声学、`solid` 固体、声固边界耦合；`std1`–`std4` 四个研究，`sol1`–`sol7` 七个求解器序列 |
| 几何 | `geom1` 在只读加载后的内存模型中运行 `geom1.run("fin")` 成功；所引用的 `loudspeaker_driver_geom_sequence.mph` 也存在于本机 COMSOL 应用示例安装中 |
| 网格 | 在同一只读加载的内存模型中运行 `mesh1.run()` 成功；批处理日志报告 7774 个域单元 |
| 已存求解场 | 对 `sol3` 的解向量读取失败；MPH 内仅有很小的 solution/mesh 条目，配套 `.m` 末尾清除了网格和所有 `sol1`–`sol7` 解数据 |
| 与仓库输入的关系 | 同目录 `.m` 与 `inputs/comsol_reference/loudspeaker_driver_exported.m` 除导出日期和 `modelPath` 外，856 行逐行相同 |

因此它是**可打开、可重建几何与网格的未求解 2D COMSOL 模型基准**。它能作为将来重建 COMSOL 计算和核对物理设置的来源；它本身不是已经求出的 `pext`、阻抗或模态参考矩阵。要对当前 Python 求解做新的数值验收，仍需在 COMSOL 中按对应研究求解并导出同口径数据。原模型的 Study 2 频率表覆盖 1–8 kHz；12–15 kHz 的 Stage35 结论不能直接由这份未修改模型复算。

本审计只读打开了 MPH 副本，没有改写原始 MPH，也没有执行完整 COMSOL 求解。仓库中的 [`benchmarks/`](../benchmarks/) 仍是精选历史结果快照，不含这个 MPH 或完整原始对照矩阵。
