# FR10 频域摇摆模态实现与验证

日期：2026-09-18  
开发分支：`feature/rocking-modes-frequency`

## 实现范围

本实现把 Cardenas 与 Klippel 摇摆模型的弱不对称前馈形式接到 FR10 full-360 cyclic 三维 FEM：

1. `k=0` 三维结构—前后声场耦合解提供单位电流音圈位移，并与 `Bl/Rdc/Le` 电路闭环得到电流和实际活塞位移；
2. `k=1` 解使用音圈体积分布的轴向载荷，经 full-360 Bloch 延拓后归一化为横向单位力矩；
3. 在完整三维音圈节点上拟合 `u_z = c + theta_x*y - theta_y*x`，得到单位力矩下的复数摇摆柔度；
4. 按论文机制分别计算质量、刚度和 Bl 力矩：

```text
mu_mass      = omega^2 Delta_m X_coil
mu_stiffness = -Delta_k X_coil
mu_Bl        = Delta_Bl_moment I
tau_cause    = H_rocking mu_cause
```

默认扰动只是诊断量级，不是 FR10 测量值：20 mg 位于 30 mm、2% `Kms` 作用于 40 mm、2% `Bl` 作用于 10 mm。

## 运行方法

```bash
python cli.py fr10-rocking \
  --config fr10_full360_cyclic/configs/rocking_modes_frequency.json \
  --outdir runs/fr10_rocking_frequency
```

输出：

- `rocking_frequency_summary.json`：完整复数结果、求解器残差和逐频验收；
- `rocking_frequency_response.csv`：适合绘图和后续辨识的平表；
- `rocking_frequency_response.png`：三类根因倾角及摇摆柔度。

## 2026-09-18 实算验证

低频验证点为 80、90、120、200、300、350、500 Hz；高频及峰值加密覆盖 750--2500 Hz。三组运行均通过自动验收：

- 最大块相对残差：`3.1945e-7`；
- 最大后向误差：`4.6737e-20`；
- 单位力矩归一化误差：`6.6613e-16`；
- 最大音圈倾角拟合相对残差：`1.9663%`。

当前 FR10 参数下，摇摆柔度从 80 Hz 的 `1.42270e-3 rad/(N m)` 缓慢上升，在已计算点中于约 2000 Hz 达到 `1.57279e-3 rad/(N m)`，随后下降。这说明代码已经产生并量化 `m=1` 摇摆响应，但该 FR10 模型的摇摆峰不在论文示例换能器的 300--350 Hz。不能通过任意改刚度把它强行调到论文频率；若要数值重合，需要对应样机的质量惯量、悬挂周向刚度及阻尼数据。

在默认诊断扰动下：

- 低频主要由刚度不对称贡献；
- 随频率升高，质量项相对增强；
- Bl 项随电流直接激励，并受电阻抗控制；
- 300 Hz 总倾角为约 `1.4108e-7 rad peak`。

上述趋势与论文的三类根因机制一致，但绝对数值只属于当前 FR10 假设参数。

## 尚未完成的物理

- 当前是弱不对称前馈模型，根因力矩不反向改变 `k=0` 活塞和电路状态；
- `k=0` 与 `k=1/k=3` 尚未装配成一个含真实缺陷分布的全耦合矩阵；
- `k=1` 是摇摆双重态的一支圆极化基，固定轴实运动需与共轭 `k=3` 组合；
- 电磁仍为等效 `Bl/Rdc/Le`，不是周向非均匀三维 MQS；
- 默认缺陷参数未经过扫描激光、Klippel 或样件测量辨识。

因此当前完成定义是“可运行、可审计的频域三维摇摆根因诊断”，不是“已验证的特定样机缺陷预测”。

## 可视化 GIF

下面的 GIF 是由 full-360 `k=1/m=1` 场生成的连续振膜动画。几何变形为观察方便进行了放大；根因 GIF 用于比较定轴、椭圆进动和圆形进动等形式，不能替代实测振幅：

- [悬挂刚度不均：80 Hz 定轴摇摆](assets/rocking_modes/01_悬挂刚度不均_80Hz_定轴摇摆.gif)
- [Bl 不均：300 Hz、45° 定轴摇摆](assets/rocking_modes/02_Bl不均_300Hz_45度定轴摇摆.gif)
- [质量偏心：2000 Hz 定轴摇摆](assets/rocking_modes/03_质量偏心_2000Hz_定轴摇摆.gif)
- [复合不对称：2000 Hz 椭圆进动](assets/rocking_modes/04_复合不对称_2000Hz_椭圆进动.gif)
- [m=1 双重态：2000 Hz 圆形进动](assets/rocking_modes/05_m1双重态_2000Hz_圆形进动.gif)
- [2000 Hz 连续振膜原始 k=1 动画](assets/rocking_modes/FR10_2000Hz_m1摇摆模态_连续振膜.gif)
- [2000 Hz 全结构节点原始 k=1 动画](assets/rocking_modes/FR10_2000Hz_m1摇摆模态_全结构节点.gif)
