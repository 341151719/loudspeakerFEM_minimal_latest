# 50 Hz 与 6300 Hz 几何和边界闭合审计

本次针对现有二维轴对称生产路由。重建命令：

```bash
python3 cli.py solve --freq 50 --drive current --current 1 --outdir runs/improvement_baseline_50Hz
python3 cli.py solve --freq 6300 --drive current --current 1 --outdir runs/improvement_baseline_6300Hz
python3 tools/audit_frequency_interfaces.py \
  --output runs/improvement_interface_audit_50_6300.json \
  --summaries runs/improvement_baseline_50Hz/summary_50Hz.json \
              runs/improvement_baseline_6300Hz/summary_6300Hz.json
```

审计以 `configs/best_model.json` 的频段路由为准。50 Hz 使用 `fast_p1`，6300 Hz 使用主配置 mapped 结构/P2 声学。两个频点共用声学网格，结构网格不同。报告逐条核对 44 个声学–结构边界的网格相邻域、旋转面积、法向及求解装配中的边界 ID；还核对 NRA 域 8/22 的存在、厚度和带标签边界，以及 Boundary93 的物理域 4/PML 域 5 相邻关系和径向法向。

| 项目 | 50 Hz | 6300 Hz |
|---|---:|---:|
| 几何审计状态 | pass | pass |
| ASB 界面边界 ID | 44 个，未漏装 | 44 个，未漏装 |
| 湿界面边段 | 171 | 171 |
| 两网格同标签旋转面积最大相对差 | 0 | 0.4027% |
| Boundary93 边段 | 104 | 104 |
| Boundary93 几何法向与径向最小点积 | 0.99943 | 0.99943 |
| NRA 域 8/22 | 各 8 个三角形；0.4/0.2 mm | 相同 |
| 求解装配最大界面投影距离 | 共形 | 0.1326 mm |
| 求解元数据 | NRA 启用；Boundary93 径向法向启用 | 相同 |

原 `production_wet_trace_audit.py` 也通过了生产湿面拓扑检查。它同时指出另一个限制：箱体参考模型的 45 mm 平面活塞与生产扬声器曲面前/后湿面不等价。本次闭合仅覆盖现有频域主线的几何与边界映射，**不**表示箱体参考活塞已可直接替代生产湿面，也不表示与 COMSOL 的全频误差已经重新验证。

机器报告与真实 FEM 输出写在 `runs/`，按仓库约定不提交。COMSOL 仍只作离线参考，本次没有读取 COMSOL 求解结果作运行时校正。
另按生产路由完成了 12000 Hz 独立求解，轴上 1 m SPL 为 92.8553 dB，输出在 `runs/improvement_baseline_12000Hz/`；该点不纳入本页低中频几何闭合结论。
