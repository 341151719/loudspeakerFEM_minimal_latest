# Sum rules FEM 离散收敛对照

参考解为结构 L2 与声学 refined1 的组合最细网格。

| 变体 | Zmot RMS | Zmot max | Rac RMS | Rac max |
|---|---:|---:|---:|---:|
| base | 0.453% | 0.722% | 5.57% | 10.4% |
| acoustic_refined | 0.333% | 0.689% | 0.856% | 2.01% |
| structure_refined | 0.32% | 0.765% | 6.25% | 12.7% |
| combined_finest | 0% | 0% | 0% | 0% |

逐频详细结果见 `frequency_comparison.csv`。
