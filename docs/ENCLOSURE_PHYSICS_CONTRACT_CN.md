# 箱体集总参考物理契约（阶段 1）

本文件固定阶段 1 的符号、单位和适用边界。六份 JSON 是“演示参数，非产品预测”；没有实测箱体、导向管或被动辐射器数据时，不得把解析曲线写成产品结论。本阶段只提供 schema 和集总参考，不生成 Gmsh 几何，不接 FEM/CLI，也不实现 BLI 或圆管 LRF PDE。

## 1. 相量、法向与功率

统一采用 `exp(+i*omega*t)`，其中 `omega=2*pi*f`。因此 `d/dt=i*omega`，惯性阻抗为 `+i*omega*M`，顺性阻抗为 `1/(i*omega*C)`，机械阻尼为 `+R`。所有场量是峰值复相量；峰值到 RMS 除以 `sqrt(2)`。峰值复功率使用

`P_avg = 0.5*Re(p*conj(U)) = 0.5*Re(F*conj(v))`。

定义 `U` 或 `v` 的正方向时，流量正方向取进入所定义网络端口；声学—结构界面法向统一为“声学域指向结构域”。作用力和反作用力必须采用同一界面、同一法向，每个湿表面只耦合一次。被动元件在该约定下满足输入阻抗实部非负；正阻尼的耗散功率非负，零损耗极限为纯无功。

## 2. 阻抗与面积变换

声学阻抗定义为 `Za = p/U`，单位 `Pa*s/m^3`；机械阻抗定义为 `Zm = F/v`，单位 `N*s/m`。对刚性活塞，`F=p*Sd`、`U=Sd*v`，所以

`Za = Zm/Sd^2`，`Zm = Za*Sd^2`。

面积只出现两次，不能把 `Zm` 直接当作 `p/U`。相同规则用于驱动振膜、PR 和箱体声学负载的机械侧换算。

## 3. 封闭箱、导向管和 PR

严格密闭、无耗背腔的空气顺性为

`C_ab = V_b/(rho0*c0^2)`，

`Z_box = 1/(i*omega*C_ab) = -i/(omega*C_ab)`，

机械侧空气弹簧为 `K_box=rho0*c0^2*Sd^2/V_b`。代码中的旧字段 `ClosedBox.loss_resistance_Pa_s_m3` 仅保留兼容性；它若有数值，会在 `p/U` 网络中加入 `1/R_leak` 并联电导，物理意义是漏气通路，不是严格密闭箱的热损耗。热黏损耗必须在独立模型中声明，不能借此字段调 Q。

倒相低频参考把箱体顺性和端口惯抗放在同一箱体压力下的并联导纳中：

`M_ap = rho0*L_eff/S_port`，

`f_b = 1/(2*pi*sqrt(M_ap*C_ab))`。

`L_eff` 的端部修正只属于该集总参考。普通两端开口管的默认第一纵向模为 open-open 半波 `c0/(2*L_eff)`；只有 API 明确指定 `closed-open`（或兼容的 `closed_open=True`）才使用四分之一波 `c0/(4*L_eff)`。若未来 FEM 几何显式解析管口、外部辐射和端部流场，必须关闭集总端部修正，并由显式外场产生辐射质量/阻尼；不得把两者再次加进 FEM PDE 或 PR/端口参数。

PR 第一版是单一刚性活塞自由度：

`(-omega^2*Mms + i*omega*Rms + 1/Cms)*x_pr = F_cavity - F_exterior`。

`v_pr=i*omega*x_pr`，`Mms/Cms/Rms` 是机械本体参数；若外部声场显式耦合，其辐射质量和辐射阻尼只能由该外场提供，不能预先重复塞入 `Mms/Rms`。缺少锥盆、悬边、材料和预张力数据时，不升级为分布式 PR FEM。

无耗、严格密闭的箱体—PR 低阶耦合调谐另行定义为

`K_box = rho0*c0^2*Sd^2/V_b`，

`f_pr,free = 1/(2*pi*sqrt(Mms*Cms))`，

`f_pr,box = sqrt((1/Cms + K_box)/Mms)/(2*pi)`。

因此 PR 自由共振与装箱调谐不是同一个数；增大 `Mms`、增大 `Cms` 或增大 `V_b` 都使装箱调谐下降，增大箱体容积同时使 `K_box` 变软。该式是无耗解析参考，不替代外场辐射阻抗。

## 4. 拓扑、损耗和阶段边界

- A `open_back` 是背腔与完整后方自由场连通，不是 pressure-release 边界；未来外场应使用闭合 HK/PML，不能沿用无限障板镜像。
- B `sealed_lossless` 是刚性不透气壁和无耗压力声学；低频 `kL << 1` 才比较均匀压力空气弹簧。
- C `sealed_thermoviscous` 保持密闭，只声明非重叠边界层 BLI 的适用性；阶段 1 不实现 BLI/LRF PDE，也不以全域复声速代替壁面损耗。
- D `vented_rear_coaxial` 只声明圆形同轴管及其圆管 LRF 适用性；管口端部修正仅用于解析参考。
- E `passive_radiator_rear_coaxial` 只声明同轴 PR SDOF；PR 与端口不能同时占据同一后置同轴安装位。

所有 JSON 的 `demonstrator` 必须为 `true`，并含来源与假设；B/C/D/E 由 `volume_contract` 明确从毛容积扣除驱动、端口或 PR 占积，使 `net_volume_target_m3` 相同。D 的管体主要向箱外伸出，背腔实际端口占积只取 `S_port*port_penetration_into_box`；为公平比较而补齐的体积必须单独记为 `fair_comparison_equalization_m3`，不能冒充实体管体。E 的 PR 背部占积必须等于 `Sd*rear_clearance`。`Mach>0.03`、位移过大、涡脱落或非线性流动时，线性结果只能标记为超出可信范围。PML 只表示出射边界的数值吸收，不得当作箱体材料耗散。
