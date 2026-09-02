# 阶段 2：轴对称 enclosure 几何合同（reference only）

本文件定义阶段 2 的可审计参考几何，不定义生产驱动接口，也不包含 Helmholtz/FEM 求解。所有图、网格和审计 JSON 均为临时输出；本阶段没有运行 solve 或 sweep。

## 1. 坐标、旋转和身份

二维坐标为 `(r,z)`，单位 m，绕 `r=0` 旋转。所有几何顶点必须满足 `r >= 0`（允许浮点误差 `1e-12`）；轴边界单独命名为 `axis`。二维面积旋转为

```text
dV = 2*pi*r*dA
dS = 2*pi*r*ds       (曲线真实旋转湿面积)
```

参考声源明确命名为 `reference planar piston`，前后面分别是 `reference_planar_piston_front` 和 `reference_planar_piston_back`。两面不共享声压自由度；它们只是 demonstrator 的参考平面活塞，不是生产驱动，也不使用旧教程 `speaker_polyline`。

参考几何的物理组 tag 是稳定的自有常量，不依赖 Gmsh 实体序号：

| 维度 | 名称 | tag |
|---|---|---:|
| 2 | `air_front_free`, `air_side_free`, `air_rear_free`, `air_cavity`, `air_rear_opening`, `air_port`, `air_pml_front`, `air_pml_rear` | 1001–1008 |
| 2 | `rigid_driver_displacement`, `rigid_comparison_equalizer`, `rigid_pr_back_mechanism` | 1009–1011 |
| 1 | `outer_pml_boundary`, `hk_front`, `hk_rear`, `axis` | 2001–2004 |
| 1 | cabinet wall、reference piston、driver cap/equalizer、opening/port/PR 接口 | 2005–2026 |

完整名称由 `src/loudspeaker_axisym_fem/enclosure_geometry.py` 的 `DOMAIN_PHYSICAL_TAGS` 与 `BOUNDARY_PHYSICAL_TAGS` 给出；审计按名称和 tag 检查，不按易变的实体编号检查。

## 2. A–E 拓扑合同

每个 case 都包含完整的前自由场、侧自由场、后自由场和闭合 HK/PML 接口。默认 `mirror=false`。PML 只覆盖外域，不能覆盖 `air_cavity` 或 `air_port`；外边界、HK、axis、cabinet wall、driver front/back、rear opening、port wall/opening、PR cavity/exterior 双面均有独立命名。

| case | 声学拓扑 | 刚体占积/接口要求 |
|---|---|---|
| A open | `air_cavity -> air_rear_opening -> air_rear_free`；后开口与完整外域连通 | driver displacement 和 comparison equalizer 为显式刚体排除体 |
| B sealed | `air_cavity` 与后自由场严格断开 | driver displacement、comparison equalizer 显式排除；净腔体积为 `0.0061 m3` |
| C sealed thermoviscous | 几何、节点、单元和物理标签与 B 完全相同 | 只允许配置损耗/适用性不同；净腔体积定义同 B |
| D rear coaxial circular port | `air_cavity -> air_port -> air_rear_free` | `air_port` 是独立空气域，单独计体积，不计入 cavity compliance；port wall/opening 不得与 driver/equalizer 相交 |
| E rear coaxial PR | `air_cavity` 与外域由 PR 界面隔开 | `pr_cavity_face` 与 `pr_exterior_face` 是分离的双面；PR 背部机构是显式刚体排除体 |

前/后 reference piston trace 始终分离；E 的 PR 两侧 trace 也分离。B/C 每一层的 mesh geometry signature 和 msh 内容相同，C 只由 thermoviscous 配置改变。所有 cavity、port、刚体排除实体由几何面实际建模，不能在报告中代数扣除。

## 3. 净体积和审计定义

`air_cavity_volume_m3` 是物理组 `air_cavity` 三角形按三角形重心积分 `2*pi*r*dA` 的离散值；目标为 `0.0061 m3`。D 的 `air_port_volume_m3` 单列。`rigid_*` 域不是空气，刚体域只用于验证显式正体积和占积位置。边-三角形邻接、域连通分量、泄漏路径、重复接口、双 trace 分离、轴边和旋转面积均由 `enclosure_topology.py` 审计。

## 4. A–E 三层结果

下表来自 15 个临时网格的串行审计（几何 signature 为确定性 hash；网格文件不提交）。所有行 `status=pass`、cavity 体积相对误差为 0（浮点舍入量级），且 `min quality >= 0.10`；D 的 L0 较低但仍为实测合格值，未放宽阈值。

| case/level | points | triangles | min quality | cavity m3 | geometry signature |
|---|---:|---:|---:|---:|---|
| A/L0 | 1384 | 2576 | 0.530554 | 0.006100 | `54a990c6…5aa15f` |
| A/L1 | 4880 | 9389 | 0.758609 | 0.006100 | `02419f5b…ae029` |
| A/L2 | 18484 | 36234 | 0.769636 | 0.006100 | `9088679d…77ba48` |
| B/L0 | 1327 | 2454 | 0.500607 | 0.006100 | `5fe10daeb…84b7` |
| B/L1 | 4815 | 9245 | 0.647616 | 0.006100 | `e3f1651e…622ea` |
| B/L2 | 18226 | 35691 | 0.769636 | 0.006100 | `4acb1fc9…5eeac` |
| C/L0 | 1327 | 2454 | 0.500607 | 0.006100 | `5fe10daeb…84b7` |
| C/L1 | 4815 | 9245 | 0.647616 | 0.006100 | `e3f1651e…622ea` |
| C/L2 | 18226 | 35691 | 0.769636 | 0.006100 | `4acb1fc9…5eeac` |
| D/L0 | 1440 | 2673 | 0.228330 | 0.006100 | `be2d7213…e39a4` |
| D/L1 | 5144 | 9890 | 0.576270 | 0.006100 | `00df5d4e…70de0` |
| D/L2 | 19354 | 37921 | 0.737479 | 0.006100 | `01b76d1e…146af` |
| E/L0 | 1388 | 2576 | 0.765989 | 0.006100 | `6f16a36a…c4afb` |
| E/L1 | 4949 | 9511 | 0.758609 | 0.006100 | `23b2ded2…4420d` |
| E/L2 | 18816 | 36865 | 0.769636 | 0.006100 | `fdb2b2ee…c088d3` |

每个 case 均满足 L0→L1→L2 全局尺寸严格减小、三角形数严格增加；B/C 每层相同。连通性为 A 后开口路径、D port 路径，B/C/E 无 cavity→exterior 路径。

## 5. 生产 wet trace（只读候选，不是 reference 接口）

来源为 `inputs/meshes/comsol_geometry_polyline_coarse_2p5mm.msh`，SHA256：
`07172d4acb374cbed2a947637decd5c9ddf67dea73553fb582bc99c6cf3d1157`。

`production_wet_trace.py` 不硬编码结果：它从每条 line physical group 的边，反查相邻三角形 physical domain；保留恰有一个 `{3,21,25}` 结构域和一个 acoustic domain `4`（front）或 `2`（rear）的共同边。每个 entity 输出 boundary/entity id、结构/声学域对、按端点连接的 `(r,z)` polyline、二维长度、`2*pi*r*ds` 真实旋转面积、径向投影面积 `2*pi*r*abs(dr)`、r/z range 和稳定 hash。

| side | boundary/entity | 结构域→声学域 | 二维长度 m | 真实 `2πr ds` m2 | 投影 `2πr|dr|` m2 |
|---|---|---|---:|---:|---:|
| front | 47 | 21→4 | 0.06153666 | 0.01631647 | 0.01262116 |
| front | 92 | 3→4 | 0.02020166 | 0.00121023 | 0.00104062 |
| front | 99 | 25→4 | 0.01253048 | 0.00542744 | 0.00351858 |
| front | 102 | 25→4 | 0.01253048 | 0.00622480 | 0.00392071 |
| rear | 46 | 21→2 | 0.06349549 | 0.01713507 | 0.01325026 |
| rear | 91 | 3→2 | 0.02032203 | 0.00122035 | 0.00104062 |
| rear | 100 | 25→2 | 0.01016817 | 0.00446555 | 0.00288948 |
| rear | 101 | 25→2 | 0.01016817 | 0.00498994 | 0.00315494 |

因此已知共同边集合为 front `47/92/99/102`、rear `46/91/100/101`。两侧汇总为：

- front：真实曲线旋转湿面积 `0.0291789398 m2`；径向投影面积 `0.0211010725 m2`，与合同参考值 `0.0211011 m2` 一致。
- rear：真实曲线旋转湿面积 `0.0278109144 m2`；径向投影面积 `0.0203353093 m2`，与合同参考值 `0.0203353 m2` 一致。

这里同时报告两种量是有意的：用户给定的历史数值精确对应投影积分，真实曲面湿面积必须使用 `2*pi*r*ds`，不能把两者混称。生产 trace 的最大半径约 `0.082 m`，不是 reference planar piston 半径 `0.045 m`。后者面积为 `pi*r^2 = 0.0063617251 m2`；与 front/rear 真实复杂湿面不等价。故 `final_production_interface_ready=false`。

## 6. 临时预览和 handoff

可读预览位于 `/tmp/luna_enclosure_phase2_plots/`：`A_L0.png`、`B_L0.png`、`C_L0.png`、`D_L0.png`、`E_L0.png` 和 `production_wet_trace.png`。人工检查未见 axis、PML、cavity、port/PR opening 的明显交叉；PNG 不进入 git。该人工预览仅供审查，不能替代 `enclosure_topology.py` 的机器连通性、邻接、体积和接口审计。

机器可读报告为 `/tmp/luna_production_wet_trace_audit.json`。阶段 handoff 是 `docs/enclosure_phase2_handoff.json`，其中记录 15 个网格的原始 SHA256、计数、质量、体积、连通性、生产输入 SHA256、湿面差异、测试命令和临时图路径。

## 7. Stage 3 入口和限制

Stage 3 只有在以下事项完成后才可进入：按真实生产运动分区映射结构湿面；对每个分区确定几何法向、位移/速度和符号；保持前/后声压 trace 及 PR 两侧 DOF 分离；验证非共形耦合的面积/法向一致性。不得用 reference planar piston 代替生产接口，不得把当前候选边界直接称作生产驱动接口。

本阶段没有求解 Helmholtz、FEM、频率响应或 sweep；`best_model`、生产 CLI、`meshgen.py` 和输入网格没有修改。
