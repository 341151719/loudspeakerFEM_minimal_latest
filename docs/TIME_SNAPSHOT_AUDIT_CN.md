# 频域代码与时域内置快照审计

频域基线：`ebc97e26f672a3aabf847617751529b55540b1d4`。时域基线：`f1c2f5b6d28d76df86bbb86c1efe4a4c7123fade`。
本清单由 `tools/audit_time_snapshot.py` 基于 Python 静态导入和文件 SHA-256 生成；动态导入及运行时行为仍需单独核对。

## 时域直接导入

- `loudspeaker_axisym_fem.axisym_magnetics`
- `loudspeaker_axisym_fem.stage4C_acoustic_structure`
- `p2_axisym_solid`

## 时域依赖闭包中的差异

- `p2_axisym_solid`：different；频域 `best_model/p2_axisym_solid.py`，时域 `inputs/frequency_mainline/best_model/p2_axisym_solid.py`。

## 逐文件对照

| 模块 | 状态 | 时域导入闭包 | 频域文件 | 时域快照文件 |
|---|---|---|---|---|
| `boundary93_parity` | same | 否 | `best_model/boundary93_parity.py` | `inputs/frequency_mainline/best_model/boundary93_parity.py` |
| `comparison` | same | 否 | `best_model/comparison.py` | `inputs/frequency_mainline/best_model/comparison.py` |
| `coupled_solver` | different | 否 | `best_model/coupled_solver.py` | `inputs/frequency_mainline/best_model/coupled_solver.py` |
| `eigenmodes` | same | 否 | `best_model/eigenmodes.py` | `inputs/frequency_mainline/best_model/eigenmodes.py` |
| `global_p2_acoustic_operator` | same | 否 | `best_model/global_p2_acoustic_operator.py` | `inputs/frequency_mainline/best_model/global_p2_acoustic_operator.py` |
| `interface_recovery` | same | 否 | `best_model/interface_recovery.py` | `inputs/frequency_mainline/best_model/interface_recovery.py` |
| `loudspeaker_axisym_fem.axisym_magnetics` | same | 是 | `src/loudspeaker_axisym_fem/axisym_magnetics.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/axisym_magnetics.py` |
| `loudspeaker_axisym_fem.axisym_solid` | same | 否 | `src/loudspeaker_axisym_fem/axisym_solid.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/axisym_solid.py` |
| `loudspeaker_axisym_fem.comsol_driver_model` | same | 否 | `src/loudspeaker_axisym_fem/comsol_driver_model.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_driver_model.py` |
| `loudspeaker_axisym_fem.comsol_feature_matrix` | same | 否 | `src/loudspeaker_axisym_fem/comsol_feature_matrix.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_feature_matrix.py` |
| `loudspeaker_axisym_fem.comsol_geom_mphtxt` | same | 否 | `src/loudspeaker_axisym_fem/comsol_geom_mphtxt.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_geom_mphtxt.py` |
| `loudspeaker_axisym_fem.comsol_mfile_inventory` | same | 否 | `src/loudspeaker_axisym_fem/comsol_mfile_inventory.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_mfile_inventory.py` |
| `loudspeaker_axisym_fem.comsol_results_compare` | same | 否 | `src/loudspeaker_axisym_fem/comsol_results_compare.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_results_compare.py` |
| `loudspeaker_axisym_fem.comsol_studies` | same | 否 | `src/loudspeaker_axisym_fem/comsol_studies.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/comsol_studies.py` |
| `loudspeaker_axisym_fem.enclosure_acoustics` | frequency_only | 否 | `src/loudspeaker_axisym_fem/enclosure_acoustics.py` | `—` |
| `loudspeaker_axisym_fem.enclosure_geometry` | frequency_only | 否 | `src/loudspeaker_axisym_fem/enclosure_geometry.py` | `—` |
| `loudspeaker_axisym_fem.enclosure_models` | different | 否 | `src/loudspeaker_axisym_fem/enclosure_models.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/enclosure_models.py` |
| `loudspeaker_axisym_fem.enclosure_schema` | frequency_only | 否 | `src/loudspeaker_axisym_fem/enclosure_schema.py` | `—` |
| `loudspeaker_axisym_fem.enclosure_topology` | frequency_only | 否 | `src/loudspeaker_axisym_fem/enclosure_topology.py` | `—` |
| `loudspeaker_axisym_fem.enclosure_validation` | frequency_only | 否 | `src/loudspeaker_axisym_fem/enclosure_validation.py` | `—` |
| `loudspeaker_axisym_fem.engineering_metrics` | same | 否 | `src/loudspeaker_axisym_fem/engineering_metrics.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/engineering_metrics.py` |
| `loudspeaker_axisym_fem.exterior_field` | same | 否 | `src/loudspeaker_axisym_fem/exterior_field.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/exterior_field.py` |
| `loudspeaker_axisym_fem.fem_solver` | same | 否 | `src/loudspeaker_axisym_fem/fem_solver.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/fem_solver.py` |
| `loudspeaker_axisym_fem.impedance_fit` | same | 否 | `src/loudspeaker_axisym_fem/impedance_fit.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/impedance_fit.py` |
| `loudspeaker_axisym_fem.json_utils` | same | 否 | `src/loudspeaker_axisym_fem/json_utils.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/json_utils.py` |
| `loudspeaker_axisym_fem.meshgen` | same | 否 | `src/loudspeaker_axisym_fem/meshgen.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/meshgen.py` |
| `loudspeaker_axisym_fem.mmcpl_lorentz_backemf` | same | 否 | `src/loudspeaker_axisym_fem/mmcpl_lorentz_backemf.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/mmcpl_lorentz_backemf.py` |
| `loudspeaker_axisym_fem.narrow_region_acoustics` | same | 是 | `src/loudspeaker_axisym_fem/narrow_region_acoustics.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/narrow_region_acoustics.py` |
| `loudspeaker_axisym_fem.postprocess` | same | 否 | `src/loudspeaker_axisym_fem/postprocess.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/postprocess.py` |
| `loudspeaker_axisym_fem.production_wet_trace` | frequency_only | 否 | `src/loudspeaker_axisym_fem/production_wet_trace.py` | `—` |
| `loudspeaker_axisym_fem.stage4B_solid_electroacoustic` | same | 是 | `src/loudspeaker_axisym_fem/stage4B_solid_electroacoustic.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4B_solid_electroacoustic.py` |
| `loudspeaker_axisym_fem.stage4C_acoustic_structure` | same | 是 | `src/loudspeaker_axisym_fem/stage4C_acoustic_structure.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4C_acoustic_structure.py` |
| `loudspeaker_axisym_fem.stage4D_exterior_nra` | same | 否 | `src/loudspeaker_axisym_fem/stage4D_exterior_nra.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4D_exterior_nra.py` |
| `loudspeaker_axisym_fem.stage4E_convergence` | same | 否 | `src/loudspeaker_axisym_fem/stage4E_convergence.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4E_convergence.py` |
| `loudspeaker_axisym_fem.stage4F_hk_refinement` | same | 否 | `src/loudspeaker_axisym_fem/stage4F_hk_refinement.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4F_hk_refinement.py` |
| `loudspeaker_axisym_fem.stage4_electroacoustic` | same | 是 | `src/loudspeaker_axisym_fem/stage4_electroacoustic.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4_electroacoustic.py` |
| `loudspeaker_axisym_fem.stage4_solid_fem` | same | 是 | `src/loudspeaker_axisym_fem/stage4_solid_fem.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/stage4_solid_fem.py` |
| `loudspeaker_axisym_fem.thermoviscous_boundaries` | frequency_only | 否 | `src/loudspeaker_axisym_fem/thermoviscous_boundaries.py` | `—` |
| `loudspeaker_axisym_fem.vibroacoustic` | same | 否 | `src/loudspeaker_axisym_fem/vibroacoustic.py` | `inputs/frequency_mainline/src/loudspeaker_axisym_fem/vibroacoustic.py` |
| `native_blocked_coil` | same | 否 | `best_model/native_blocked_coil.py` | `inputs/frequency_mainline/best_model/native_blocked_coil.py` |
| `p2_axisym_solid` | different | 是 | `best_model/p2_axisym_solid.py` | `inputs/frequency_mainline/best_model/p2_axisym_solid.py` |
| `p2_pml_operator` | same | 否 | `best_model/p2_pml_operator.py` | `inputs/frequency_mainline/best_model/p2_pml_operator.py` |
| `sweep_checkpoint` | frequency_only | 否 | `best_model/sweep_checkpoint.py` | `—` |
| `visualization` | different | 否 | `best_model/visualization.py` | `inputs/frequency_mainline/best_model/visualization.py` |

同步规则：仅当频域改动落在时域导入闭包内，才逐文件审查并在时域仓库单独提交；不得覆盖整个快照目录。
函数级人工核对和时域基线记录见[时域仓库说明](https://github.com/341151719/loudspeakerTimeFEM_minimal_latest/blob/main/docs/FREQUENCY_SNAPSHOT_AUDIT_CN.md)。
