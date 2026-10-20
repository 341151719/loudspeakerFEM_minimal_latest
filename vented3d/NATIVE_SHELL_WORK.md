# Native shell/interface implementation work

User request (2026-10-20): implement all three missing mechanisms in Python: MITC shell interpolation, fold-line director DOFs/constraints, and finite-thickness/offset solid–shell coupling. Preserve the completed baseline and its provenance. Current branch: `feature/vented3d-native-shell-interfaces`, based on `7cb7c1d`.

Confirmed local settings in `inputs/vented_enclosure/live_settings.csv`: `shell/prop/ShellAdvancedSettings.UseMITCInterp=1`; `shell/prop/FoldlineAngle.FoldlineAngle=0.001`; pointwise elemental fold constraints; `coupling/sshc1.connectionSettings=sharedBnd`, `connectionType=Hard`, shell-thickness distance, no offset in thickness features. Shell translations/directors are quadratic: tri6 / quad9. Solid displacement is tet10 / pyr13 / hex20. Existing Python shells use faceted P2/mean shear and shared axial rotations; connector uses solid curl.

Implementation status: all three mechanisms are connected to `build_native_coupled` under `configs/native_shell_mitc.json`. Corrected trial04 has completed independent 315/3550 Hz regression, source 5da57f9; the completed acoustic baseline has not been overwritten. Real-model preflight checks are performed before starting a fresh 3550/315 Hz run. Unsupported geometry or constraints fail explicitly.

Implementation details:

1. `src/vented3d/mitc_shell.py`: curved tri6/quad9 director shells and MITC tensor tying. MITC6-a shear uses eight coefficients (two Gauss points on each edge plus two center components); membrane tying uses nine samples. MITC9 covariant 11/13 components use 2x3 tying grids, 22/23 use 3x2, 12 uses 2x2. Geometry and directors are interpolated consistently so every rigid rotation has zero strain. Plane-stress stiffness and consistent mass are integrated through the physical thickness. This is not the previous mean-shear stabilization with a renamed switch.
2. `shell_topology.py` and `shell_constraints.py`: director tangency, native smooth-boundary ownership, separate fold directors and linearized fixed-angle constraints. Director is normal displacement `a`, not axial rotation `theta`. Shared translations remain continuous. Opposite normals receive a signed director relation. No artificial drilling stiffness is used. The actual geometry contains 2033 native shells (1273 tri6, 760 quad9), 6995 director nodes and 809 fold-node groups. Director cuts come directly from native arx/ary/arz DOF ownership, under the exported 0.001-radian setting; they are not guessed from planar facets.
3. `native_solid_connections.py`: branch-specific native solid–shell connections. The actual source selects **sharedBnd** and **NoOffset**: displacement traces coincide, and shell directors stay independent. Native quad9 traces on hex20/pyr13 faces are evaluated with the solid shape functions. Only the 12 shared boundary tags are connected; incidental edge/point contacts add no ties. `Hard` and `distanceEvaluation=ShellProperties` are saved properties of other branches and do not prescribe thickness-fiber director constraints here. A separate finite-thickness plane-section operator and integral hard mortar remain implemented and tested for suitable connection geometries; they are not forced onto this shared-face model.
4. `native_shell_model.py`: actual assembly into the native acoustic/motor/BEM route, with updated pressure work maps and constraint reduction. The dedicated config enables all three mechanisms together. `shell_settings.py` checks the corresponding physical export settings. Tests include curved rigid motions, membrane/bending patches, a thickness range down to 1e-5, absence of extra flat-element zero modes, actual fold topology and offset force/moment transfer. Numerical full-model and COMSOL accuracy gates remain separate.

5. `native_solids.py`: the source's solid order `2s` is implemented separately as tet10/pyr13/hex20, with curved geometry, consistent mass, native stiffness and motor integration. This is necessary to connect native curved shells without extrapolating into the former straight-solid mesh. The dataset contains 12059 solid elements and 29635 displacement nodes (88905 scalar displacement DOFs, versus 316005 before conversion was removed). Shared shell/solid geometry agrees to 2.8e-17 m. Missing entries belonging to other geometry orders are never indexed as valid nodes.

The published MITC6a/MITC9 variants and exact elemental constraint enforcement are not certified identical to COMSOL's internal implementation. BEM discretization, shell/solid convergence and exact correspondence of connection enforcement remain evidence limits until controlled full-model checks pass. No material constants or reference-result gains have been fitted.

Primary references:

- COMSOL 6.3 shell theory: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.130.html
- COMSOL fold settings: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_shell_plate.08.02.html
- COMSOL connection: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_multiphysics.18.74.html
- Lee & Bathe 2004: https://web.mit.edu/kjb/www/Principal_Publications/Development_of_MITC_Isotropic_Triangular_Shell_Finite_Elements.pdf
- Kim & Bathe 2009: https://web.mit.edu/kjb/www/Principal_Publications/A_Triangular_Six-Node_Shell_Element.pdf
- MITC9: https://web.mit.edu/kjb/www/Principal_Publications/Towards_Improving_the_MITC9_Shell_Element.pdf

Offline diagnostics already created in the migrated project, `continuation_20261019/diagnosis_3550/`: response attribution and native parametric surface motion. At 3550 Hz current error 1.076%, velocity/current 20.365%; 74-direction radiation error 22.593%, after coil normalization 5.548%. Composite boundaries 200/202 dominate surface displacement error. Native pressure control with reference motion is ~0.1%, but independent full pressure error is 34.89%. These are diagnostics, not production gain corrections.

Other settings audited: damping reference 40 Hz, thickness and quadratic order agree. Exported `massMatrix=lumped` is under StructuralTransientBehavior and its applicability to the frequency-domain study is not established; do not switch Python mass blindly. Study geometric nonlinearity is off.

## Shared-boundary branch correction

The initial trial03 incorrectly added normal-fiber slope constraints on shared faces. Its numerical residuals passed but both 315 and 3550 Hz responses were almost suppressed (about 100% complex error). It is a rejected experiment, not an accepted implementation. A mechanical-only 315 Hz control reused identical K/M/loss/C matrices: full fiber ties produced coil velocity per ampere 1.48e-7 + 3.14e-6j m/s/A; removing the inapplicable director ties restored 0.02123 - 0.19780j m/s/A. Keeping only tangential fiber ties was still suppressed; removing fold-angle constraints alone did not restore the response.

The official theory distinguishes shared cladding faces from shell edges perpendicular to a solid. `distanceEvaluation` belongs to the parallel-boundary branch; `Hard`/`Rigid` does not activate the perpendicular plane-section law on shared faces. The corrected production route uses `native_shared_boundary`. Tests explicitly allow a solid normal strain and independent shell directors at a shared face, while still enforcing all translation traces. No response calibration is involved.

## Corrected independent regression

| Hz | 线圈速度复误差：原基线 → 本次 | +X 声压复误差：原基线 → 本次 | +Z 声压复误差：原基线 → 本次 | 内部压力 DOF L2：原基线 → 本次 |
|---:|---:|---:|---:|---:|
|315|0.890% → 0.122%|1.102% → 1.028%|0.908% → 0.688%|1.589% → 0.213%|
|3550|20.694% → 0.876%|23.398% → 1.730%|26.831% → 0.962%|34.890% → 2.255%|

Both frequencies pass numerical residual checks. These are independent solves; references enter only the offline comparator. The corrected route has not passed a uniform 1% gate or a fresh 46-point sweep. See benchmarks/native_shell_regression.json and native_shell_regression_provenance.json.

## Current acceptance policy (user update)

From 2026-10-20 the future reproduction accuracy target is 3%, replacing the earlier 1% target. Existing independent 315/3550 Hz results pass all six recorded metrics at 3%; maximum errors are 1.028% and 2.255%. Historical statements above describe the original goal. Full repaired 46-point coverage, modal matching and convergence remain open. No new calculation is needed to apply this policy to saved evidence.
