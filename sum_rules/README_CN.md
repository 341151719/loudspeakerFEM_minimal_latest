# Loudspeaker FEM SUM RULES 计算与验证

本目录收纳 SUM RULES 的固定模型计算方法、机器可读数据、图表和分析报告。生产扫频会按频率切换离散配置；SUM RULES 审计始终使用同一个固定配置，因而全频曲线对应单一个离散线性系统。

## 已验证到的程度

- 0.1 Hz–15 kHz 的已算频点上，运动阻抗和辐射后内部损耗余量均为非负，支持被动性与功率分解。
- 1 Hz–15 kHz 的低频加权矩 M2 达到实静态端点 A0 的 99.534%。
- 该工作频带的 M0 为 20,324.73 ohm/s。321 点结果经 66 个自适应频点加密后只变化 0.104%，后验局部积分差指标为 1.54%。
- 15 kHz 内的 M0 是矩阵高频端点 Ainf 的 35.798%。Ainf 包含超出有效工作带的高阶结构自由度，因此工程判读以局部带宽积分和其收敛率为主。
- 有限带 M0 中，有用辐射效率项占 4.177%，内部损耗惩罚项占 95.823%。
- 复 `Zmot` 的六锚点空间离散差异低于约 0.8%。仅声学加密后，远场 `Rac` RMS 差异从 5.57% 降至 0.856%。

完整解读见 [`report/FINAL_ANALYSIS_CN.md`](report/FINAL_ANALYSIS_CN.md)。

材料参数稳健性另用锥盆面密度和杨氏模量的 0.5×/1×/2× 扫描验证。五种模型的 M2/A0 为 99.48%–100.33%，并全部满足被动性；局部 M0 会随模态位置明显重排。结果与方法见 [`results/parameter_scan/REPORT_CN.md`](results/parameter_scan/REPORT_CN.md)。

## 计算定义

反映到电端口的运动阻抗为

```text
Zmot = e_back / I
```

远场辐射电阻、内部损耗余量和效率定义为

```text
Rac       = 2 P_rad / |I|^2
Rinternal = Re(Zmot) - Rac
eta       = Rac / (Re(Zblocked) + Re(Zmot))
```

两个有限频窗积分为

```text
M0 = (2/pi) integral Re(Zmot) d(omega)
M2 = (2/pi) integral Re(Zmot) / omega^2 d(omega)
```

效率预算按下式分解：

```text
Re(Zmot) = Re(Zblocked) * eta/(1-eta) + Rinternal/(1-eta)
```

对实际工作频带，使用局部带宽量

```text
M0[w1,w2] = (2/pi) integral(w1,w2) Re(Zmot) d(omega)
```

或使用以 `s` 为频带中心尺度的 Poisson/Stieltjes 权重：

```text
(2s/pi) integral(w1,w2) Re(Zmot)/(omega^2+s^2) d(omega) <= Zmot(i s)
```

建议取 `s = sqrt(w1*w2)`。这个量可直接用于局部效率–带宽预期，无需把超高频矩阵自由度当作工程带宽。

## 固定配置

主配置是 [`../configs/sum_rules_highest_accuracy.json`](../configs/sum_rules_highest_accuracy.json)，组合：

- 结构域 21/25 的 L2 局部加密网格；
- 声学域 1/2/4/5/7/8/22 的 refined1 网格；
- 声学域 2/4/7 的选择性 P2；
- 固定开启原生 NRA；
- `exp(+i omega t)` 相量约定。

离散规模为 244,862 个自由结构自由度和 171,649 个声学未知量。输入网格、静磁偏置场和配置的 SHA-256 记录在 [`results/main_321/sum_rule_summary.json`](results/main_321/sum_rule_summary.json)。

## 复算方法

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

运行 321 点固定模型扫描：

```bash
python tools/validate_sum_rules_fem.py \
  --freqs log:1:15000:321 \
  --outdir runs/sum_rules_highest_accuracy \
  --jobs 5 --blas-threads 1 --resume
```

在内存约 24 GiB 的机器上，5 个并行进程是经过实测的安全设置；单工作进程在稀疏分解期间可达约 3.3 GiB。

生成自适应频点：

```bash
python tools/select_sum_rule_adaptive_frequencies.py \
  runs/sum_rules_highest_accuracy/sum_rule_frequency_audit.csv \
  runs/sum_rules_adaptive_round1/frequencies.csv \
  --coverage 0.99

python tools/validate_sum_rules_fem.py \
  --freqs runs/sum_rules_adaptive_round1/frequencies.csv \
  --outdir runs/sum_rules_adaptive_round1 \
  --jobs 5 --blas-threads 1 --resume

python tools/merge_sum_rule_adaptive_audit.py
```

自适应子集的频率不连续，不能单独跨区间积分；必须使用合并脚本将它与 321 点主网格结合。

网格对照使用 50 Hz、1/4/8/12/15 kHz 六个锚点，分别运行：

```text
configs/stage35_high_accuracy.json
configs/stage35_high_accuracy_acoustic_refined1.json
configs/stage35_high_accuracy_local_l2.json
configs/sum_rules_highest_accuracy.json
```

最后运行：

```bash
python tools/compare_sum_rules_discretizations.py
```

材料参数扫描使用以下四个配置；`sum_rules_highest_accuracy.json` 是共享的 1× 基准：

```text
configs/sum_rules_cone_density_0p5.json
configs/sum_rules_cone_density_2p0.json
configs/sum_rules_cone_stiffness_0p5.json
configs/sum_rules_cone_stiffness_2p0.json
```

每个变体先运行 81 点主网格，再用 `select_sum_rule_adaptive_frequencies.py --coverage 0.95` 选点、用 `merge_sum_rule_adaptive_audit.py` 合并。统一汇总命令是：

```bash
python tools/analyze_sum_rule_parameter_scan.py
```

具体命令、频点和输入哈希保存在 `results/parameter_scan/raw/` 内各案例的 JSON 文件中。

## 数据与报告

- [`results/main_321/`](results/main_321/)：321 点完整主扫描、累积积分图和功率审计图。
- [`results/adaptive_387/`](results/adaptive_387/)：自适应选点清单、66 点原始子集，以及合并后的 387 点数据和收敛结果。
- [`results/mesh_convergence/`](results/mesh_convergence/)：2×2 结构/声学网格锚点对照；`raw/` 保留四个变体的原始 CSV/JSON。
- [`results/low_frequency/`](results/low_frequency/)：0.1/0.2/0.5/1 Hz 低频端点数据。
- [`results/parameter_scan/`](results/parameter_scan/)：锥盆面密度和刚性的 0.5×/1×/2× 扫描、局部带宽积分、原始数据与专项报告。
- [`report/FINAL_ANALYSIS_CN.md`](report/FINAL_ANALYSIS_CN.md)：综合分析、结论等级与后续路径。

CSV 是主要科学数据，JSON 保留配置、输入哈希、端点、检查项和运行命令，PNG 只是对应 CSV 的可视化。

## 解读边界

当前材料包含频率无关的结构 loss factor。它适合现有工程 FEM 预测，但外推到严格零频时会产生很小的 M2 对数项。因此当前结果对有限工作带是定量验证；若要建立严格 0–无穷 Herglotz 证明，需要将其替换为因果粘弹性模型。
