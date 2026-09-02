# 阶段 4B：密闭 C 参考箱体热黏边界合同

## 范围与身份

本阶段只把阶段 4A 的 fixed/isothermal/no-slip thermoviscous boundary-layer impedance（BLI）接入密闭 C 参考几何。声源仍是半径 0.045 m、全局 `v_z=+1 m/s` 的 **reference planar piston**。结果是 demonstrator/reference，不是生产驱动或产品 SPL；`final_production_interface_ready=false`。

B 与 C 使用阶段 2 已审计且几何完全相同的网格。C 只增加边界损耗；不使用全域复声速，不修改正式配置、生产网格、CLI 或旧求解器。

## +iωt 弱式

体域采用

\[
\int_\Omega 2\pi r\left[\rho^{-1}\nabla p\cdot\nabla q
-\omega^2K_s^{-1}pq\right]dA.
\]

COMSOL 给出的 inward-normal BLI 条件为

\[
-\mathbf n\cdot\mathbf v=-i\omega
(T_{bnd}-\alpha_pTp/(\rho C_p))\delta_t\alpha_p/(1+i)
-v_n+\delta_v(\nabla_t\cdot\mathbf v_{0t}+\Delta_t p/(i\omega\rho))/(1+i).
\]

固定、等温、无滑移壁令 `T_bnd=v_n=v_0t=0`，并用理想气体恒等式
`α_p²T/(ρCp)=(γ-1)/K_s`。将压力相关项移到弱式左端、对切向 Laplacian 做表面分部积分，得到

\[
\int_{\Gamma_{BLI}}2\pi r
\left[c_v\nabla_t p\cdot\nabla_t q+c_tpq\right]ds,
\]

其中

\[
\delta_v=\sqrt{2\mu/(\rho\omega)},\qquad
\delta_t=\sqrt{2\kappa/(\rho C_p\omega)},
\]

\[
c_v=-\delta_v/[\rho(1+i)],\qquad
c_t=-\omega^2(\gamma-1)\delta_t/[K_s(1+i)].
\]

在 `exp(+iωt)` 下两系数虚部非负。对 peak phasor，物理边界耗散为

\[
P_v=\operatorname{Im}(c_v)\,p^HK_tp/(2\omega),\quad
P_t=\operatorname{Im}(c_t)\,p^HM_bp/(2\omega),\quad
P_{BLI}=P_v+P_t\ge0.
\]

`K_t`、`M_b` 与 FEM 矩阵均由同一 `FacetBasis` 几何积分生成，但功率通过独立低层二次型交叉核对。`loss_scale=0` 时 BLI 矩阵严格为零，B/C 同网格压力场逐 DOF 相同。

## 边界选择合同

只选择恰有一个压力邻域且该域为 `air_cavity` 的 physical line facet；每条边必须恰有一个 physical owner，并排除 axis、pressure-pressure、HK、PML、外场边界及 piston front。当前选择：

- `cabinet_front_wall`
- `cabinet_side_wall`
- `comparison_equalizer_face`
- `driver_side_wall`
- `reference_planar_piston_back`

后者既保留 prescribed normal velocity source，又施加 fixed/isothermal/no-slip BLI；两者代表同一运动壁的法向源和切向/热边界修正，不重复压力 DOF。L1/L2 旋转总面积均为 `0.197720741960685 m²`，piston back 面积为 `π·0.045² m²`。

## 收敛离散

阶段 2 CAD 与 `.msh` 不变。压力 P1 网格在内存中对上述腔壁相邻单元做三次确定性 conforming red/green/blue refinement：父三角形 physical domain 由 subdomain 映射传播；每条命名 facet 必须保持 1→1 或严格分裂为 1→2。细化后再次检查 tag、pressure component、HK/PML、体积、面积和 piston trace/Q。

| 层级 | 压力 DOF | 三角形 | BLI facets | cavity volume/m³ |
|---|---:|---:|---:|---:|
| L1 | 6761 | 12519 | 704 | 0.006100000000000 |
| L2 | 22022 | 42065 | 1392 | 0.006100000000000 |

`Q_front/Q_back=-/+0.006361725123519 m³/s`；局部加密前后体积、BLI面积与trace积分在浮点容差内不变。

## 功率与数值证据

主功率恒等式是

\[
P_{in}=P_{PML,numerical}+P_{BLI,physical}.
\]

PML 是开放外域的数值吸收，不是材料耗散；HK outward flux仅作独立交叉检查。10/100/500/1000 Hz 的所有 `Pvisc`、`Pthermal`、PML吸收和输入功率均为正，最大主闭合相对残差为 `2.01e-8`。

L1→L2 raw `Ptotal`差分别为 `0.0127% / 1.4770% / 4.8209% / 2.1055%`，不改变 `<5%`门槛；最大阻抗幅值差 `1.46e-6 dB`，最大HK轴上SPL差 `0.1189 dB`。500 Hz：

| 层级 | Pvisc/W | Pthermal/W | Ptotal/W |
|---|---:|---:|---:|
| L1 | 6.483291455e-4 | 1.559884235e-4 | 8.043175690e-4 |
| L2 | 6.890159212e-4 | 1.560414062e-4 | 8.450573274e-4 |

10 Hz L2 曾出现 combined matrix 与独立分项二次型相差 `2.5903e-9 W`，而主功率闭合相对残差仅 `2.004e-8`。交叉核对因此采用有量纲、scale-aware 判据
`|err| <= 5e-9 W + 1e-6*max(|Ptotal|,|Pin|,|PML|,|Pindependent|)`；它不参与物理功率计算或网格收敛判据。

局部第一腔模扫描中，B sampled peak约 `876.5 Hz`且采样带宽未解析；C peak约 `876.0 Hz`，峰值由约 `2.283e5 Pa`降至`2.708e4 Pa`，采样半功率带宽约`1.25 Hz`、Q约`700.8`。这只证明参考声腔峰值被BLI降低，不代表箱体机械模态或产品Q。

## 限制与后续入口

- BLI适用于声明的10–1000 Hz与平滑壁段；尖角邻域尚未做full-TV局部验证。
- 没有热黏体域、结构耦合、生产湿面速度映射或产品预测。
- 当前 production wet trace 与 reference piston 不等价；后续必须映射真实运动分区、法向和速度。
- 未做A开放箱热黏、D port、E PR或大范围参数扫描。
- 下一阶段只能在保留raw功率、网格和接口审计的前提下扩展，不得把PML数值吸收称为物理耗散。

