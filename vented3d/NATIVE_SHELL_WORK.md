# Native shell/interface implementation work

User request (2026-10-20): implement all three missing mechanisms in Python: MITC shell interpolation, fold-line director DOFs/constraints, and finite-thickness/offset solid–shell coupling. Preserve the completed baseline and its provenance. Current branch: `feature/vented3d-native-shell-interfaces`, based on `7cb7c1d`.

Confirmed local settings in `inputs/vented_enclosure/live_settings.csv`: `shell/prop/ShellAdvancedSettings.UseMITCInterp=1`; `shell/prop/FoldlineAngle.FoldlineAngle=0.001`; pointwise elemental fold constraints; `coupling/sshc1.connectionSettings=sharedBnd`, `connectionType=Hard`, shell-thickness distance, no offset in thickness features. Shell translations/directors are quadratic: tri6 / quad9. Solid displacement is tet10 / pyr13 / hex20. Existing Python shells use faceted P2/mean shear and shared axial rotations; connector uses solid curl.

Implementation status: all three mechanisms are connected to `build_native_coupled` under `configs/native_shell_mitc.json`. The first 49 route tests passed. A fresh two-point run (3550 then 315 Hz) is being assembled in a separate local output directory. Full-model accuracy is pending; the completed acoustic baseline has not been overwritten.

Implementation details:

1. `src/vented3d/mitc_shell.py`: curved tri6/quad9 director shells and MITC tensor tying. MITC6-a shear uses eight coefficients (two Gauss points on each edge plus two center components); membrane tying uses nine samples. MITC9 covariant 11/13 components use 2x3 tying grids, 22/23 use 3x2, 12 uses 2x2. Geometry and directors are interpolated consistently so every rigid rotation has zero strain. Plane-stress stiffness and consistent mass are integrated through the physical thickness. This is not the previous mean-shear stabilization with a renamed switch.
2. `shell_topology.py` and `shell_constraints.py`: director tangency, smooth-boundary merging, separate fold directors and linearized fixed-angle constraints. Director is normal displacement `a`, not axial rotation `theta`. Shared translations remain continuous. Opposite normals receive a signed director relation. No artificial drilling stiffness is used. The actual geometry contains 2033 native shells (1273 tri6, 760 quad9) and 919 fold-node groups.
3. `native_connections.py`: `u_solid = u_reference + z*a`, using bounded solid traces over the shell thickness and offset. Translation and director are reconstructed from zeroth/first thickness moments at connected nodes. This transfers force and moment and replaces the solid-curl tie. A reusable integral hard-mortar operator is also provided in `shell_constraints.py`. Missing thickness coverage, disconnected layers or extrapolated ownership raise errors. Existing straight P2 solid geometry is retained and explicitly reported.
4. `native_shell_model.py`: actual assembly into the native acoustic/motor/BEM route, with updated pressure work maps and constraint reduction. The dedicated config enables all three mechanisms together. `shell_settings.py` checks the corresponding physical export settings. Tests include curved rigid motions, membrane/bending patches, a thickness range down to 1e-5, absence of extra flat-element zero modes, actual fold topology and offset force/moment transfer. Numerical full-model and COMSOL accuracy gates remain separate.

The published MITC6a/MITC9 variants and the hard thickness projection are not certified identical to COMSOL's internal implementation. The unmodified solid geometry, BEM discretization, shell convergence and exact correspondence of connection enforcement remain evidence limits until controlled full-model checks pass. No material constants or reference-result gains have been fitted.

Primary references:

- COMSOL 6.3 shell theory: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.130.html
- COMSOL fold settings: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_shell_plate.08.02.html
- COMSOL connection: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_multiphysics.18.74.html
- Lee & Bathe 2004: https://web.mit.edu/kjb/www/Principal_Publications/Development_of_MITC_Isotropic_Triangular_Shell_Finite_Elements.pdf
- Kim & Bathe 2009: https://web.mit.edu/kjb/www/Principal_Publications/A_Triangular_Six-Node_Shell_Element.pdf
- MITC9: https://web.mit.edu/kjb/www/Principal_Publications/Towards_Improving_the_MITC9_Shell_Element.pdf

Offline diagnostics already created in the migrated project, `continuation_20261019/diagnosis_3550/`: response attribution and native parametric surface motion. At 3550 Hz current error 1.076%, velocity/current 20.365%; 74-direction radiation error 22.593%, after coil normalization 5.548%. Composite boundaries 200/202 dominate surface displacement error. Native pressure control with reference motion is ~0.1%, but independent full pressure error is 34.89%. These are diagnostics, not production gain corrections.

Other settings audited: damping reference 40 Hz, thickness and quadratic order agree. Exported `massMatrix=lumped` is under StructuralTransientBehavior and its applicability to the frequency-domain study is not established; do not switch Python mass blindly. Study geometric nonlinearity is off.
