# 扬声器背腔、导向管与被动辐射器 FEM：Luna 分阶段实施规划

文档状态：实施前规划，尚未授权修改生产模型
制定日期：2026-08-25
上游仓库：https://github.com/341151719/loudspeakerFEM_minimal_latest
上游基线：`main@99deff739cb977d85af1a202fcd9d37376ced803`
Luna 执行配置：所有阶段固定使用 `gpt-5.6-luna`，`reasoning_effort=max`
目标项目：二维轴对称、小信号、频域、电磁—结构—声学耦合 FEM
相量约定：`exp(+i*omega*t)`；出射 Green 函数采用 `exp(-i*k*R)`

## 0. 文档用途与执行原则

本文件是后续调用 Luna 的唯一总任务合同。Luna 不应一次性完成全部工作，而应严格按阶段执行；每个阶段只接收必要文件、前一阶段短交接和本阶段验收条件。只有当前阶段的硬门槛全部通过，才进入下一阶段。

规划优先级如下：

1. 物理定义正确，符号、单位、功率和边界方向可审计；
2. 几何是真正可旋转成三维实体的二维轴对称几何；
3. 从解析极限、集总模型、网格收敛和能量守恒四条独立证据验证；
4. 不破坏现有 `configs/best_model.json` 生产链；
5. 用分阶段上下文、短报告和可复用脚本节省 Luna token；
6. 在没有实测箱体/导向管/被动辐射器参数时，只输出“演示模型”，不得宣称产品预测。

## 1. 需求冻结与最终对比矩阵

需要建立五种相互可比的模型：

| ID | 模型 | 严格物理定义 | 主要输出 |
|---|---|---|---|
| A | 开放背腔 | 同一箱体轮廓，后壁存在开放口，振膜背面经背腔直接连通自由场；不是把背腔压力强制设为零 | 电阻抗、振膜运动、前/后辐射、低频偶极抵消 |
| B | 封闭背腔 | 同一净容积、刚性不透气壁面、腔内空气默认无耗压力声学 | 箱体空气弹簧、阻抗峰、腔模、位移抑制 |
| C | 封闭热耗散背腔 | 与 B 相同且无质量泄漏；在适用条件内加入热黏边界层耗散。若另有吸声棉，则作为 C2 独立子模型，不与热黏壁损混称 | 热/黏耗散功率、Q 值变化、被动性和能量平衡 |
| D | 导向管/倒相箱 | 封闭腔体加真实后置同轴圆形导向管，导向管连通后方自由场；管内热黏损耗按圆管模型处理 | Helmholtz 调谐、管口流速/体积速度、双阻抗峰、前后辐射合成 |
| E | 被动辐射器 | 封闭腔体后壁安装同轴圆形被动辐射器；第一版为刚性活塞单自由度 `Mms/Cms/Rms`，两侧均与 FEM 声场耦合 | PR 位移、调谐、驱动振膜位移谷、双阻抗峰、PR 辐射 |

公平比较必须满足：

- 使用同一驱动单元、电磁力、blocked coil、结构材料和驱动电压；另给 1 A 峰值电流归一化结果以隔离电气负载变化。
- B/C/D/E 使用相同目标净背腔容积；D 的导向管占积、E 的 PR 背部占积必须从毛容积扣除。
- 使用同一外部箱体轮廓、观察点、空气性质、网格层级和远场定义。
- A 的后开口是唯一有意变化；D/E 的开口或运动部件是各自唯一拓扑变化。
- 任何为匹配曲线而逐频调整的材料、阻尼、端部修正或增益都禁止进入生产结果。

## 2. 已完成的项目审计结论

### 2.1 源码状态

- GitHub 默认分支为 `main`，审计基线 commit 是 `99deff739cb977d85af1a202fcd9d37376ced803`。
- 当前本地工作副本不含 `.git` 元数据。与临时浅克隆逐文件比较后，现有源码、配置和输入与该 commit 相同；本地只缺远端的 `.gitignore` 和 `README.md`。
- 后续推荐在新的完整克隆中创建 `feature/enclosure-axisym-fem` 分支开发。当前目录只作为审计参考和规划文档位置，不进行高风险原地改造。
- 当前 Python 环境缺少 `meshio` 与 `scikit-fem`，所以本次规划阶段未完成自检和网格审计。用户已明确授权配置全部缺失环境；这不是项目阻塞，但环境复现仍是 Luna 阶段 0 的硬门槛。

### 2.2 可复用能力

- `src/loudspeaker_axisym_fem/enclosure_models.py` 已有 `ClosedBox`、`Port`、`VentedBox`、`PassiveRadiator` 和 `PassiveRadiatorBox` 集总模型，可作为解析/低阶基准，不能直接称为 FEM 结果。
- `src/loudspeaker_axisym_fem/meshgen.py` 和 `fem_solver.py` 已有独立的简化开/闭背腔网格和压力声学教程链，但几何、标签和求解器没有接入当前 `best_model` 生产装配。
- `best_model/coupled_solver.py` 已实现 P2 结构、非共形 ASB、blocked coil、电压/电流驱动、PML/HK 外场和频域耦合，是最终集成目标。
- `src/loudspeaker_axisym_fem/narrow_region_acoustics.py` 已有平行板狭缝热黏等效模型和耗散功率函数。导向管是圆管，禁止直接复用狭缝公式。
- 当前生产声学域和边界依赖 COMSOL `mphtxt` 的固定 domain/boundary 编号。新箱体不应继续向这些硬编码集合堆叠编号，而应建立独立、命名化的 enclosure topology contract。

### 2.3 不得破坏的生产合同

- `configs/best_model.json` 不做探索性原地修改。
- 原有 `solve/sweep/blocked/eigen` 命令行为、P1/P2 频率路由、Boundary 93、NRA 8/22、Lorentz/back-EMF 共轭约定保持不变。
- 新功能使用独立配置和独立 CLI 子命令；只有全矩阵回归通过后，才讨论是否合并为默认入口。
- COMSOL 数据只允许作为离线 benchmark，不允许变成运行时校正表。

## 3. 物理模型合同

### 3.1 适用范围

第一目标频段为 10–1000 Hz，用于箱体、导向管和 PR 的特征行为；第二目标频段扩展至 3 kHz，用于背腔模态和过渡趋势。现有生产链仍需在 50、6300、12000 Hz 做不变性回归，但不能因此宣称新轴对称箱体在 12 kHz 已经物理收敛。

全部结果是线性小信号频域结果。至少输出并检查：

- 驱动振膜与 PR 最大位移相对特征尺寸足够小；
- 导向管峰值速度、`Mach = |v|/c0` 和位移幅值；
- `Mach > 0.03`、明显涡脱落或高声压非线性风险时，标记该点超出线性模型可信范围；
- 不使用线性 FEM 预测喘流、喷流分离、端口压缩或大位移悬边非线性。

### 3.2 轴对称几何限制

二维 `(r,z)` 网格旋转后必须形成真实三维实体。因此：

- 基准导向管采用后壁中心的同轴圆管；其截面为 `0 <= r <= a_port`，沿负 `z` 方向伸出箱体。
- 基准 PR 采用后壁中心的同轴圆盘。
- 偏心圆管、侧置管、矩形管、矩形箱和任意非旋转对称 PR 不得直接在二维轴对称中声称为真实几何。
- 如果产品必须使用偏心圆管，可增加“等效环形管”研究，但必须标为等效模型，并同时匹配截面积、声学质量和湿周/水力直径；最终产品结论应升级至 3D。

### 3.3 压力声学

无耗空气域采用频域压力声学，弱式保留轴对称体积权重 `2*pi*r`：

`div((1/rho_eff) grad(p)) + omega^2/K_eff * p = 0`。

无耗空气取 `rho_eff=rho0`、`K_eff=rho0*c0^2`。全部单位采用 SI，场变量为峰值复相量；SPL 转 RMS 时除以 `sqrt(2)`。

### 3.4 声固耦合

驱动结构保持现有生产式：

`(K_complex - omega^2 M) u - G p = F_Lorentz`
`A_ac p - rho0*omega^2 G^T u = 0`

耦合矩阵 `G` 必须由同一物理界面上的边积分得到；法向统一由声学域指向结构域。每个湿表面只耦合一次，不允许因重合边界重复加载。

### 3.5 开放背腔

开放背腔是连通声学域，不是 pressure-release 边界。前、后声场在完整自由场 PML/HK 外边界内共同求解。因为后辐射存在，`mirror_sound_hard_plane=true` 的无限障板镜像不能直接沿用；新模型应使用完整轴对称球形/截球形外域和闭合 HK 面，默认 `mirror=false`。

低频验证应观察到：

- 驱动振膜前后体积速度大小接近、相位相反；
- 相比封闭/障板情况出现与声程差相关的低频抵消；
- 结果随外域半径、PML 厚度和后开口网格收敛。

### 3.6 封闭无耗背腔

壁面为刚性不透气边界。低频、腔内 `kL << 1` 时，FEM 应收敛到均匀压力空气弹簧：

`C_ab = V_b/(rho0*c0^2)`，
`Z_box = 1/(i*omega*C_ab)`，
`K_box,mechanical = rho0*c0^2*S_d^2/V_b`。

FEM 净容积必须通过轴对称积分计算，不能只用 CAD 标称值。腔内均匀压力极限、输入声阻抗和解析式应分别比较。

### 3.7 封闭热耗散背腔

“热耗散”主模型 C1 定义为壁面附近热黏边界层造成的耗散，且背腔仍严格密闭。选型规则：

- 边界层不重叠，且局部曲率半径和间隙远大于 `delta_v/delta_t`：使用热黏边界层阻抗（BLI）或等价弱式边界算子；
- 长直、恒定或缓变窄通道，横向尺寸与边界层可比且低于横向截止：使用对应截面的 Narrow Region/LRF 等效流体；
- 边界层重叠且几何急变、曲率很强：局部升级为完整热黏声学，不能强套 BLI；
- 大体积空气默认仍是绝热压力声学。不得用一个无来源的全域复声速假装壁面热耗散。

所有热黏模型必须在 `exp(+i*omega*t)` 约定下通过被动性测试：耗散功率非负、声学输入阻抗实部不产生能量、关闭耗散参数后连续回到模型 B。

若实际需求中的“热耗散”指吸声棉/多孔衬层，则建立 C2：单独给出流阻率、孔隙率、曲折度、黏性/热特征长度或明确的经验模型适用范围。C1 与 C2 分开报告，禁止用调 Q 的单一虚数参数替代材料卡。

### 3.8 导向管

导向管是实际 FEM 空气域，包含管内驻波、入口/出口辐射和外场耦合。解析低频基准为：

`M_ap = rho0*L_eff/S_port`，
`f_b = 1/(2*pi*sqrt(M_ap*C_ab)) = c0/(2*pi)*sqrt(S_port/(V_b*L_eff))`。

集总基准可使用端部修正；真实 FEM 几何已经显式包含入口/出口和外部辐射时，不允许再次把相同端部修正加进 PDE，以免双重计数。

管损耗必须使用圆管 LRF/宽管模型或经验证的 BLI。每个频率记录：`delta_v/a`、`delta_t/a`、`k*a`、第一纵向管模频率和模型适用性。导向管有明显截面变化或喇叭口时，先用 BLI；只有恒截面段才能使用圆管等效域参数。

### 3.9 被动辐射器

第一版 PR 使用一个物理上明确的刚性活塞自由度 `x_pr`，不是逐节点互不相干的局部阻抗。其方程为：

`(-omega^2*Mms + i*omega*Rms + 1/Cms) x_pr = F_cavity - F_exterior`。

PR 两侧压力通过面积积分形成净力，`v_pr=i*omega*x_pr` 再作为两侧声学法向速度；作用力与速度必须互为功率共轭。需要额外包含由 FEM 外场自动产生的辐射质量和辐射阻尼，因此不得在 `Mms/Rms` 中重复加入同一辐射项。

只有当目标频段接近 PR 膜/锥盆 breakup，且已有厚度、密度、弹性、泊松比、预张力、悬边几何和阻尼数据时，才升级到分布式 PR 结构 FEM。缺少这些参数时，SDOF 刚性活塞比虚构柔性材料更真实。

## 4. 参数与配置合同

新配置放在 `configs/enclosures/`，由一个基础配置和五个变体组成，不覆盖现有配置：

```text
configs/enclosures/base_axisym.json
configs/enclosures/open_back.json
configs/enclosures/sealed_lossless.json
configs/enclosures/sealed_thermoviscous.json
configs/enclosures/vented_rear_coaxial.json
configs/enclosures/passive_radiator_rear_coaxial.json
```

基础配置至少包含：

- `provenance`：上游 commit、配置版本、生成命令、参数来源和是否实测；
- `geometry`：箱体内/外尺寸、壁厚、后开口、驱动安装面、外域和 PML；
- `net_volume_target_m3` 与各部件占积明细；
- `port`：半径、直管长度、喇叭口几何、表面粗糙度假设和损耗模型；
- `passive_radiator`：`Sd_m2/Mms_kg/Cms_m_N/Rms_N_s_m` 及参数来源；
- `thermoviscous`：空气热物性、BLI/NRA 选择和适用性阈值；
- `mesh`：L0/L1/L2 全局和局部尺寸、元素阶次、PML 设置；
- `study`：驱动模式、频率段、峰值附近自适应加密规则；
- `limits`：线性速度、位移、Mach 和可信频段。

参数校验必须拒绝：负体积/质量/顺性/阻尼、零截面积、导向管与驱动几何相交、PR 面积超过后壁、PML 内半径穿过实体、损耗模型与截面类型不匹配、没有单位来源的裸值。

在用户尚未提供实物参数时，可创建 `demonstrator=true` 的演示配置，但报告标题、JSON metadata 和图标题都必须显式写“演示参数，非产品预测”。

## 5. 建议代码边界

最终文件名可在阶段 1 小幅调整，但职责不得混杂：

```text
src/loudspeaker_axisym_fem/
  enclosure_schema.py              # 配置、单位、参数和适用性校验
  enclosure_geometry.py            # 确定性 Gmsh 轴对称几何/物理组
  enclosure_topology.py            # 域、边界、连通性、法向和净容积审计
  enclosure_acoustics.py           # 新声学算子、完整外域/PML/HK
  thermoviscous_boundaries.py       # BLI 与圆管 LRF，功率和被动性
  passive_radiator_coupling.py      # PR 刚性活塞 SDOF 耦合
  enclosure_solver.py              # 规定速度、生产驱动集成和结果对象
  enclosure_validation.py          # 解析极限、能量、收敛和比较指标

tools/
  enclosure_mesh_audit.py
  enclosure_compare_cases.py

tests/
  test_enclosure_schema.py
  test_enclosure_geometry.py
  test_enclosure_lumped_limits.py
  test_enclosure_thermoviscous.py
  test_enclosure_port.py
  test_enclosure_passive_radiator.py
  test_enclosure_energy.py
  test_enclosure_regression.py
```

CLI 建议增加隔离命令组，例如：

```text
python cli.py enclosure-mesh --config ... --level L0 --outdir ...
python cli.py enclosure-check --config ... --mesh ... --outdir ...
python cli.py enclosure-solve --config ... --freq 50 --drive current ...
python cli.py enclosure-sweep --cases ... --freqs ... --jobs 1 ...
python cli.py enclosure-compare --run-root ...
```

生产 `solve/sweep` 不得静默改变语义。探索结果写入 `runs/enclosure_study/`，不写入 `inputs/`，不提交 VTU/NPZ/PNG。

## 6. Luna 分阶段执行计划

### 阶段 0：Git、环境和原始基线

目标：得到可回滚、可运行且结果可复现的工作分支。

操作：

1. 从上游重新克隆，确认 HEAD 等于 `99deff739cb977d85af1a202fcd9d37376ced803`；创建 `feature/enclosure-axisym-fem`。
2. 安装 Python 3.11+ 虚拟环境和锁定依赖版本；记录 Python、NumPy、SciPy、meshio、scikit-fem、gmsh、BLAS 信息。用户已授权安装缺失组件，但应优先使用项目隔离虚拟环境，不为本项目降级或覆盖系统公共 Python 包。
3. 执行 `python cli.py self-test`、完整 pytest 和原始 50 Hz 求解；资源允许时执行 6300/12000 Hz。
4. 记录原始命令、耗时、峰值内存、输出摘要和关键输入 SHA-256。

硬门槛：

- HEAD 正确，工作树初始干净；
- self-test 与现有测试全通过；
- 50 Hz 原始真实 FEM 成功；
- 失败时只修环境，不修改物理源码；无法复现则停止。

Luna 本阶段上下文只需：`README.md`、`README_CN.md`、`requirements.txt`、`pyproject.toml`、`cli.py`。

### 阶段 0P：设备性能与并行校准

目标：在不改变物理和数值离散的前提下，找出当前设备的可靠吞吐配置；此阶段位于阶段 0 与阶段 1 之间。

已知硬件基线：WSL2 可见 18 个逻辑 CPU、23 GiB 内存和 16 GiB swap。阶段 0 的 50 Hz 单频 `splu` 求解耗时约 13.89 s、峰值 RSS 约 402 MiB；依赖安装的主要瓶颈是 `/mnt/c` 的 9p/DrvFS 小文件 I/O，而不是下载带宽。

操作：

1. 不重装现有 `.venv`；比较 `/mnt/c` 与 WSL ext4 临时目录的模型装载、输出和小型求解 I/O。
2. 用相同低频配置和 8 个互不相同频点，测量 `jobs=1/2/4/6/8`、`blas_threads=1` 的总耗时、单点耗时、峰值总内存和结果一致性。
3. 对单个代表频率只做 `blas_threads=1/2/4` 的短基准，确认 OpenBLAS 线程对 SciPy SuperLU 是否实际有益；不得凭 CPU 占用猜测。
4. 初始运行策略为：低频扫频 6–8 个频率进程、每进程 1 BLAS 线程；中高频按实测内存降为 2–4 个进程。
5. 可只读评估多线程稀疏后端（例如 MKL/PARDISO）的可用性、许可证、安装体积和预期收益；阶段 0P 不把它接入生产。若未来引入，必须另做复残差、解 NRMSE、功率和 50/6300/12000 Hz 回归。
6. 运行暂存和高频 scratch 优先放 WSL ext4，最终机器数据再复制到项目 `runs/`；复制后核对 SHA-256。源码仍留在 Git 工作树，禁止建立两个可同时修改的源码副本。

硬门槛：

- 各 worker 数结果与串行复数结果在舍入容差内一致；
- 选出的 worker 数以墙钟吞吐和内存共同决定，不以 CPU 占用率单独决定；
- 峰值总内存保留至少 25% 余量，不依赖 swap 才能完成；
- 不修改物理源码、配置值或离散阶次来制造速度提升；
- 输出 `docs/enclosure_phase0p_performance.json`，记录推荐的低/中/高频 jobs、BLAS 线程和 scratch 位置。

Luna 本阶段上下文只需：阶段 0 handoff、本节、`cli.py` 中 sweep/worker 控制、`best_model/coupled_solver.py` 的求解入口。

### 阶段 1：物理/拓扑合同与集总参考

目标：在写新 FEM 前固定参数 schema、解析基准、符号和功率约定。

操作：

1. 审计 `enclosure_models.py` 的单位、网络拓扑、端部修正和 PR 方程；先写测试，再修正发现的问题。
2. 新建配置 schema 和五个演示变体；所有参数带来源和 `demonstrator` 标志。
3. 实现 sealed/vented/PR 的解析扫频参考及特征频率计算。
4. 写一页“相量、法向、峰值/RMS、机械/声学阻抗转换”合同。

硬门槛：

- 解析模型维度检查通过；
- 零损耗极限为纯无功，正阻尼下平均耗散非负；
- 端部修正只能出现在集总参考，不能污染未来 FEM PDE；
- PR 自振频率和 vented Helmholtz 频率有独立手算测试。

### 阶段 2：参数化几何与纯网格审计

目标：只证明几何和物理组有效，不求解多物理。

操作：

1. 从生产驱动湿表面提取可追溯的轴对称轮廓；不要使用与生产轮廓不一致的教程 speaker polyline 作为最终接口。
2. 生成 A–E 五种拓扑，使用命名物理组，不依赖易变的 Gmsh 实体序号。
3. 输出 L0 网格、域/边界编号图、轴对称旋转示意、净容积和连通分量报告。
4. 增加几何相交、孤立域、非流形边、重复界面、负半径、退化单元和法向审计。

硬门槛：

- 每个预期声学域连通关系与模型定义一致；
- sealed/C/PR 腔体无泄漏；vented 只有导向管连通；open 只有后开口连通；
- 轴对称积分体积与 CAD/解析体积误差小于 0.5%；
- 驱动和 PR 湿表面面积误差小于 0.5%；
- L0/L1/L2 标签集合完全一致；
- 任何几何失败时不得进入求解阶段。

### 阶段 3：规定振膜速度的 A/B 声学 FEM

目标：在不引入电磁和结构复杂度前，验证开放/封闭声学及完整外域。

操作：

1. 对驱动湿面施加规定的单位法向速度，求 A 和 B。
2. B 在多个低频点提取腔内平均压力和驱动面输入声阻抗，与 `V/(rho*c^2)` 解析极限比较。
3. A 检查前后体积速度、低频抵消和完整远场。
4. 对外域半径、PML 厚度/强度、HK 面、L0/L1/L2 做单变量扫描。

硬门槛：

- B 在 `kL <= 0.2` 区域，平均压力/输入阻抗相对解析解：L1 小于 2%，L2 小于 1%；
- B 无耗模型的物理耗散接近数值零，不能出现显著负阻；
- A 的外域/PML 改变导致轴上幅值小于 0.2 dB、相位小于 2°；
- 耦合面体积速度积分与规定输入误差小于 0.5%；
- 网格加密趋势单调或有可解释的高阶收敛。

### 阶段 4：C 封闭热耗散背腔

目标：加入有适用范围、可计算功率且保证被动的热黏耗散。

操作：

1. 实现边界层厚度和 BLI 适用性报告；不满足条件时自动拒绝或路由到 NRA/完整热黏方案。
2. 先用解析平面/圆管测试验证弱式符号，再用于背腔壁。
3. 输出黏性、热耗散和总耗散功率；比较 B/C 共振频率、峰值和 Q。
4. 令黏度/热导率趋零或关闭模型，验证连续回到 B。

硬门槛：

- 所有测试频率耗散功率 `>= -数值容差`；
- 关闭耗散后 B/C 复场 NRMSE 小于 0.2%；
- L1/L2 耗散功率变化小于 5%，关键响应小于 0.3 dB；
- 总输入平均功率与物理耗散之差小于 2%（极低功率点改用绝对容差）。

### 阶段 5：D 导向管 FEM

目标：获得真实导向管调谐、管内损耗和管口辐射结果。

操作：

1. 先求无耗圆管，再加圆管 LRF/BLI；不得调用现有 slit 模型冒充圆管。
2. 用解析 Helmholtz 网络预测 `f_b`，围绕预测值自适应加密频率。
3. 输出管内压力/速度幅相、入口/出口体积速度、管口辐射功率、驱动振膜位移和电阻抗。
4. 扫描导向管长度、半径和网格；确认调谐变化方向符合解析尺度律。

硬门槛：

- 无耗低频 FEM 调谐相对包含一致端部定义的解析值：L1 小于 5%，L2 小于 3%；
- 增大 `L` 使调谐下降，增大 `S` 使调谐上升，增大 `V` 使调谐下降；
- 调谐附近驱动振膜位移出现谷、管口体积速度出现峰；电压驱动总阻抗呈可解释双峰；
- 圆管损耗开启后峰值降低且耗散非负，不能造成增益；
- 端口 `Mach` 超限的点在结果中自动标红并排除线性可信结论。

### 阶段 6：E 被动辐射器耦合

目标：实现全局刚性活塞 SDOF PR 与腔内/外 FEM 声场的双向耦合。

操作：

1. 先对独立 PR SDOF 做解析自由振动和受力响应测试。
2. 将一个 PR 位移 DOF 以面积积分耦合到两侧压力，检查矩阵互易/功率共轭。
3. 扫描 `Mms/Cms/Rms` 并与被动辐射箱集总网络比较。
4. 输出 PR 位移/速度、两侧平均压力、净声力、辐射功率和悬挂耗散。

硬门槛：

- 真空/无声场下 PR 自振频率误差小于 0.2%；
- 加入声场后，频移方向与附加辐射质量一致；
- 增大 `Mms` 降低调谐，增大刚度提高调谐，增大 `Rms` 降低 Q；
- 调谐附近 PR 位移峰与驱动振膜位移谷同时出现；
- PR 机械耗散 `0.5*Rms*|v|^2` 非负，耦合功率残差小于 2%。

### 阶段 7：接入现有电磁—结构生产驱动

目标：把已经独立验证的 enclosure 声学载荷接到现有 P2 驱动和 blocked coil，而不是重写生产求解器。

操作：

1. 复用生产结构、Lorentz、back-EMF 和 blocked impedance；替换/扩展声学拓扑装配层。
2. 使用非共形 P2 ASB 将生产湿表面与新声学网格耦合；做刚体位移和单位压力补丁测试。
3. 同时实现 current drive 与 voltage drive；检查总阻抗 `Z_total=Z_blocked+Z_motional` 的符号。
4. 对五个模型完成低频代表点，再做 50/6300/12000 Hz 原生产回归。

硬门槛：

- `enclosure=none` 或原始配置路径的结果与阶段 0 基线在数值容差内一致；
- Lorentz/back-EMF 功率互易误差不恶化；
- current drive 下结构/声学单位载荷缩放线性；
- voltage drive 下电气输入功率与线圈、结构、热黏、PR 和辐射功率闭合；
- 原有测试全部通过，新测试不得通过 monkeypatch 隐藏物理错误。

### 阶段 8：扫频、网格收敛和五模型对比

目标：产出可追溯的 FEM 结果和可信范围，不只给图片。

扫频策略：

1. 10–1000 Hz 先用稀疏对数网格定位特征点；
2. 在阻抗峰/谷、位移峰/谷、Helmholtz/PR 调谐和腔模附近自动细化；
3. 扩展到 3 kHz；
4. 每个关键点执行 L0/L1/L2，不要求每个探索频率都跑最细网格；
5. 只在低/中阶段通过后运行高成本全场图和高频回归。

并行参数必须来自阶段 0P 的实测推荐；每个频率进程默认保持 1 个 BLAS 线程，避免 worker×BLAS 嵌套超卖。若模型规模变化使峰值内存超过阶段 0P 预测，应自动降低 jobs，而不是使用 swap 硬撑。

最终至少报告：

- 总电阻抗实部/虚部/幅值/相位；
- 电流、Lorentz 力、驱动振膜平均与最大位移/速度；
- PR 位移/速度或导向管入口/出口体积速度；
- 背腔平均压力、压力非均匀度和声学模态；
- 1 m 轴上 SPL/相位、全角指向性、前/后/port/PR 分量；
- 辐射功率、线圈损耗、结构损耗、热/黏损耗、PR 损耗、PML 吸收和功率残差；
- `delta_v`、`delta_t`、`k*a`、Mach、网格 DOF、耗时、内存；
- L0/L1/L2 变化与 Richardson/观测阶次（数据允许时）。

硬门槛：见第 7 节总验收矩阵。任何模型未过门槛时，报告为“诊断结果”，不进入最终对比结论。

### 阶段 9：独立 benchmark 与交付

目标：在 Python 自验证成立后，选择性增加 COMSOL 或实测对照。

- COMSOL 使用独立几何和同一参数表，输出原始复数阻抗、复声压、位移和功率；不在 Python 运行时读取为校正值。
- 有实物时优先测量自由空气/封闭/倒相或 PR 电阻抗，以及近场管口/振膜响应。
- 交付源码、配置、网格生成器、测试、CSV/JSON/NPZ、图、运行命令、环境信息、输入哈希和限制说明。
- 结果二进制留在 `runs/` 或发布附件，不提交到最小源码包。

## 7. 总验收矩阵

| 类别 | 指标 | 通过标准 |
|---|---|---|
| 几何 | 域连通、密闭/开口拓扑 | 与 A–E 定义完全一致，无孤立/重复域 |
| 几何 | 净容积、湿面积 | 与解析/CAD 值误差 `<0.5%` |
| 网格 | L1→L2 关键幅值 | 一般 `<0.3 dB`；调谐频率 `<2%` |
| 网格 | L1→L2 关键相位 | 一般 `<3°`，深零点单独解释 |
| PML/HK | 外域/PML 参数变化 | 轴上 `<0.2 dB`、`<2°` |
| sealed | 低频空气弹簧 | L1 `<2%`、L2 `<1%` |
| thermal | 耗散/阻抗被动性 | 所有物理耗散非负，无负阻发能 |
| vented | Helmholtz 调谐 | L1 `<5%`、L2 `<3%`，趋势律正确 |
| PR | 自由 SDOF 自振 | `<0.2%`；耦合调谐趋势正确 |
| 耦合 | 体积速度/界面积分 | 相对误差 `<0.5%` |
| 能量 | 输入功率闭合 | 一般 `<2%`；共振尖峰/极小功率点 `<5%` 或绝对容差 |
| 回归 | 原生产 50/6300/12000 Hz | 未启用新功能时与基线数值等价 |
| 线性 | 端口/PR/驱动适用性 | 超限点明确标记且不进入可信结论 |
| 可追溯 | 配置、commit、哈希、命令 | 每个结果文件均可反查 |

阈值是初始工程验收线，不是测量精度声明。如果解析模型本身不满足低频假设，禁止用该解析误差作为 FEM 失败判据，应先记录无量纲适用性。

## 8. 结果与机器数据合同

每次 solve 目录至少包含：

```text
run_manifest.json
geometry_audit.json
mesh_metrics.json
summary_<freq>.json
solution_<freq>.npz
solid_<freq>.vtu
acoustic_<freq>.vtu
boundary_flux_<freq>.csv
power_balance_<freq>.json
applicability_<freq>.json
```

每次 sweep 至少包含：

```text
sweep_metrics.csv
case_comparison.csv
mesh_convergence.csv
resonance_features.json
run_manifest.json
```

`run_manifest.json` 必须记录：Git commit、dirty 状态、配置路径及 SHA-256、输入文件哈希、Python/包版本、操作系统、命令、相量约定、峰值/RMS 约定、开始/结束时间、耗时、线程数和模型可信标志。

所有图必须由机器数据重建，标题注明模型 ID、频率、驱动、网格层、单位、相量约定和 `demonstrator/product` 状态。

## 9. 安全、回滚与禁止事项

### 9.1 源码安全

- Luna 只在完整 Git 克隆的功能分支工作；每阶段一个小 commit。
- 阶段开始前工作树必须干净；阶段结束提交前输出 `git diff --stat` 和测试清单。
- 不使用 `git reset --hard`、不覆盖上游输入、不删除用户文件。
- 不把探索网格/结果放入 `inputs/`；最终输入网格只有在生成过程确定、来源完整并通过清洁测试后才考虑提交。
- 每个阶段可独立回退，不允许把多个未验证物理改动压在同一 commit。

### 9.2 物理安全

- 禁止错误相量符号导致的负耗散；每种阻尼都必须有功率测试。
- 禁止把 PML 吸收当作箱体材料耗散；PML 只代表出射功率数值边界。
- 禁止 FEM 与集总网络同时计入同一端部修正、辐射质量或辐射阻尼。
- 禁止把 slit NRA 用于圆管、把 BLI 用于重叠边界层、把均匀复声速用于无来源的壁面热损。
- 禁止把 2D 轴对称结果外推到偏心管/矩形箱的三维局部流动和指向性。
- 禁止在 `Mach`、位移或涡脱落超出小信号条件时继续声称线性预测有效。
- 禁止仅凭单频曲线“看起来合理”升级；必须同时有极限、能量和网格证据。

### 9.3 停止条件

出现以下任一情况，Luna 必须停止当前物理扩展，仅提交诊断：

- 负耗散或总输入功率明显不闭合；
- 几何连通/法向不唯一；
- L1/L2 不收敛且 PML/外域变化同量级；
- 解析极限在其适用区间仍不匹配；
- 新功能关闭后原生产结果发生不可解释变化；
- 需要缺失的实物参数才能作出关键选择；
- 运行资源超过约定上限或求解器出现不受控内存增长。

## 10. Luna token 节省协议

### 10.1 每阶段上下文包

每次调用只提供：

1. 本文档对应阶段；
2. 上一阶段 `handoff.json`；
3. 最多 4–6 个直接相关源码文件；
4. 失败测试的完整日志或成功测试的一行摘要；
5. 当前 `git diff --stat`。

不要重复发送整个仓库、完整历史工具目录、全部 COMSOL 导出或所有运行结果。需要搜索时先用 `rg` 定位，再读取命中片段。

所有阶段均使用 `gpt-5.6-luna` 且推理强度固定为 `max`；不得为了速度在中途降为 medium/low。token 节省依靠缩小上下文和简化回复实现，不通过降低推理强度实现。

### 10.2 Luna 输出限制

Luna 每阶段回复只需：

- 结论：通过/未通过/阻塞；
- 修改文件列表和每个文件一句用途；
- 执行的测试与结果；
- 物理验收数值；
- 未解决风险；
- 下一阶段所需的最小上下文。

不要复述本规划，不要粘贴大段源码；代码通过 Git diff 交付。建议文字报告控制在约 600–1000 中文字，失败时保留关键堆栈。

### 10.3 阶段交接格式

每阶段生成短 `handoff.json`：

```json
{
  "phase": 0,
  "status": "pass|fail|blocked",
  "commit": "...",
  "changed_files": [],
  "tests": [{"command": "...", "result": "pass"}],
  "physics_gates": [{"name": "...", "value": 0.0, "limit": "...", "result": "pass"}],
  "known_risks": [],
  "next_files": []
}
```

`handoff.json` 不替代详细机器结果，只用于减少下一次模型上下文。

### 10.4 Luna 单阶段提示词模板

```text
你只执行《LUNA_ENCLOSURE_FEM_PLAN_CN.md》的阶段 N。
基线 commit：<commit>；上一阶段状态：<handoff>。
允许读取：<files>。允许修改：<files/dirs>。
禁止修改 configs/best_model.json、原生产输入和阶段外文件。
先运行阶段前测试，再实施；只做一个物理变量层。
必须完成本阶段硬门槛；失败则停止并提交诊断，不进入下一阶段。
最终仅返回：状态、diff stat、测试、物理门槛数值、风险、下一阶段最小文件列表。
```

## 11. 推荐的首次 Luna 调用

第一次只执行阶段 0，不写物理代码。推荐提示：

```text
执行 docs/LUNA_ENCLOSURE_FEM_PLAN_CN.md 的阶段 0。
从 https://github.com/341151719/loudspeakerFEM_minimal_latest 的
main@99deff739cb977d85af1a202fcd9d37376ced803 创建独立工作克隆和
feature/enclosure-axisym-fem 分支。复现 Python 环境，运行 self-test、完整 pytest、
原始 50 Hz FEM，并记录环境、资源、哈希和 handoff.json。
本阶段禁止修改任何物理源码和 configs/best_model.json；失败只诊断环境。
输出遵守第 10 节，不开始阶段 1。
```

## 12. 用户参数检查点

阶段 2 以前可以使用明确标注的演示参数。进入最终产品计算前，需要用户提供或确认：

- 实际箱体内尺寸、壁厚、后开口和驱动安装位置；
- 目标净容积及驱动、管、PR、支撑件占积；
- 导向管内径、物理长度、两端倒角/喇叭口和表面情况；
- PR 的 `Sd/Mms/Cms/Rms`，以及这些值是否已经包含空气附加质量；
- 热耗散究竟是裸刚性壁热黏损、吸声衬层还是全填充材料；
- 吸声材料参数及来源；
- 驱动电压/功率、目标频段和最大允许位移/端口速度；
- 是否有实测阻抗、近场振膜/管口或 SPL 数据用于独立验证。

这些参数未确认前，可以验证方法和趋势，不能发布绝对性能结论。

## 13. 完成定义

只有同时满足下列条件，需求才算真正完成：

1. A–E 五种几何、配置、源码和测试均自包含；
2. 规定速度声学基准、集总解析极限和生产多物理链均通过；
3. 每个耗散项被动，完整功率平衡闭合；
4. 关键频率完成 L0/L1/L2 和外域/PML 收敛；
5. 原生产模型在新功能关闭时无回归；
6. 五模型在相同净容积/驱动/观察条件下完成对比；
7. 结果包含阻抗、位移、声压、指向性、功率、损耗、适用性和资源数据；
8. 所有结论标注二维轴对称、线性小信号、频段和参数来源限制；
9. COMSOL/实测若参与，只作独立 benchmark，不作运行时校正；
10. 交付可由一组记录的命令从干净克隆重建。

## 14. 物理参考入口

- 项目上游及生产合同：https://github.com/341151719/loudspeakerFEM_minimal_latest
- COMSOL Pressure Acoustics 基本假设（无耗、绝热、等熵）：https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_introduction.02.04.html
- Thermoviscous Boundary Layer Impedance 的适用条件：https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_pressure.05.034.html
- Narrow Region/LRF 的 slit、圆管和宽管模型及适用范围：https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_pressure.05.186.html
- 官方 vented loudspeaker enclosure 的声固耦合与内/外声场建模示例：https://doc.comsol.com/6.4/doc/com.comsol.help.models.aco.vented_loudspeaker_enclosure/vented_loudspeaker_enclosure.html

这些参考用于核对方程、适用性和 benchmark 设计，不授权复制 COMSOL 结果或在 Python 运行时依赖 COMSOL。
